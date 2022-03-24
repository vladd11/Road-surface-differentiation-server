OUTPUT_PATH = '/home/rozhk/output'
# PYTHONASYNCIODEBUG = True

import asyncio
import time

import cv2
import imutils
import numpy
import openrouteservice as openrouteservice
from fastai.vision import *
from websockets import serve

from color import colorfull_fast, color_roads

TARGET = (float(sys.argv[2]), float(sys.argv[1]))
ROUTING_KEY = os.environ.get('ROUTING_KEY')

torch.set_num_threads(2)
learner = load_learner('/home/rozhk')
client = openrouteservice.Client(key=ROUTING_KEY)


def predict(frame):
    # noinspection PyTypeChecker
    img = Image(pil2tensor(frame, np.float32).div_(255))
    predicted = learner.predict(img)[0]

    predicted_width = predicted.shape[1] - 1
    predicted_height = predicted.shape[2] - 1

    return predicted.data.numpy(), predicted_width, predicted_height


def get_center(prediction):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(color_roads(prediction.astype(
        dtype=np.uint8)[0]), cv2.MORPH_OPEN, kernel, iterations=5)

    contours = cv2.findContours(
        opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    cv2.imwrite('test.png', opening)

    temp = []
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        temp.append((int(M["m10"] / M["m00"]) / prediction.shape[2], int(M["m01"] / M["m00"]) / prediction.shape[1]))

    return temp


async def echo(websocket):
    async for message in websocket:
        if not isinstance(message, bytes):
            message = message.split(' ')
            message = message[1:3]
            for i, e in enumerate(message):
                message[i] = float(e)

            routes = client.directions((message, TARGET),
                                       optimize_waypoints=True, format='geojson')
            await websocket.send('route ' + json.dumps(routes['features'][0]['geometry']['coordinates'][1:]))
            await websocket.send('img')
        else:
            await websocket.send('img')
            prediction, width, height = predict(
                cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR))

            cv2.imwrite(os.path.join(OUTPUT_PATH, f'{time.time()}.png'),
                        colorfull_fast(prediction[0], numpy.empty(dtype=numpy.uint8, shape=(width, height, 3)), width,
                                       height))
            await websocket.send(json.dumps(get_center(prediction)))


async def main():
    async with serve(echo, "0.0.0.0", 8765):
        await asyncio.Future()


asyncio.run(main())
