OUTPUT_PATH = '/home/rozhk/output'
VIDEO_LATENCY = 10  # In seconds

import asyncio
from websockets import serve
import cv2
import numpy
import openrouteservice as openrouteservice
from fastai.vision import *

from color import colorfull_fast
from reader import VideoCapture

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "err_detect;aggressive|" \
                                              "fflagsx;nobuffer|" \
                                              "flags;low_delay|" \
                                              "framedrop|" \
                                              "fflags;strict|" \
                                              "protocol_whitelist;file,rtp,udp"

learner = load_learner('/home/rozhk')

cam = VideoCapture("stream.sdp")
client = openrouteservice.Client(key=ROUTING_KEY)


def predict(frame_to_predict):
    # noinspection PyTypeChecker
    img = Image(pil2tensor(frame_to_predict, np.float32).div_(255))
    predicted = learner.predict(img)[0]

    predicted_width = predicted.shape[1] - 1
    predicted_height = predicted.shape[2] - 1

    return predicted.data.numpy(), predicted_width, predicted_height


shouldSendFrame = False
imagePosValues = ()


async def echo(websocket):
    async for message in websocket:
        print('test')


async def main():
    async with serve(echo, "localhost", 8765):
        await asyncio.Future()


asyncio.run(main())

'''class Handler(BaseRequestHandler):
    def handle(self):
        ready = False
        buffer = []
        first = False

        global shouldSendFrame
        global imagePosValues

        while True:
            if first and not ready:
                self.request.sendall(b'\x80')
                ready = True
            first = True
            packet = self.request.recv(1024)
            if ready:
                if packet == b'j':
                    ready = False
                    prediction, width, height = predict(
                        cv2.imdecode(np.frombuffer(bytes(buffer), np.uint8), cv2.IMREAD_COLOR))
                    cv2.imwrite('test.png', colorfull_fast(prediction[0], numpy.empty(
                        dtype=numpy.uint8, shape=(width, height, 3)), width, height))

                    buffer = []
                else:
                    buffer += packet
            else:
                if packet == b"b":
                    while not imagePosValues:
                        shouldSendFrame = True

                    if imagePosValues is None:
                        print("imagePosValues is null")
                        return

                    # means road is straight
                    self.request.sendall(struct.pack('%sf' % len(imagePosValues), *imagePosValues))
                    imagePosValues = ()
                elif packet.startswith(b"r"):
                    json_packet = json.loads(packet[1:])
                    routes = client.directions((json_packet[u'c'], json_packet[u'e']),
                                               optimize_waypoints=True, format='geojson')
                    print(json.dumps(routes['features'][0]['geometry']['coordinates'][1:]))
                    self.request.sendall(
                        json.dumps(routes['features'][0]['geometry']['coordinates'][1:]).encode("utf-8") + b'\n')
                elif packet == b"c":
                    return


def run():
    with TCPServer(("0.0.0.0", 1071), Handler) as server:
        server.serve_forever()
'''

'''class ImageHandler(BaseRequestHandler):
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)
        self.buffer = None

    def handle(self):
        self.request.settimeout(1000)
        while True:
            try:
                packet = self.request.recv(1024).strip()
            except socket.timeout:
                if self.buffer != None:
                    prediction, width, height = predict(cv2.imread())

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    opening = cv2.morphologyEx(color_roads(prediction.astype(
                        dtype=np.uint8)[0]), cv2.MORPH_OPEN, kernel, iterations=5)

                    contours = cv2.findContours(
                        opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = imutils.grab_contours(contours)

                    cv2.imwrite('test.png', opening)

                    if len(contours) == 1:
                        M = cv2.moments(contours[0])
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        imagePosValues = (cX / prediction.shape[2], cY / prediction.shape[1])
                    else:
                        temp = []
                        for c in contours:
                            # compute the center of the contour
                            M = cv2.moments(c)
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])

                            print(cX / prediction.shape[2], cY / prediction.shape[1])

                            temp.append(cX / prediction.shape[2])
                            temp.append(cY / prediction.shape[1])
                        imagePosValues = temp

                    shouldSendFrame = False
                self.buffer = None
            else:
                self.buffer += packet

while True:
    frame = cam.read()
    if shouldSendFrame:
        print('Receiving last frame')
        sleep(VIDEO_LATENCY)  # TODO: Bad way to get round the latency

        
    elif frame is not None:
        prediction, width, height = predict(frame)
        # because it works well
        # noinspection PyTypeChecker
        cv2.imwrite(os.path.join(OUTPUT_PATH, f'{round(time() * 1000)}.png'), colorfull_fast(prediction[0], numpy.empty(
            dtype=numpy.uint8, shape=(width, height, 3)), width, height))
'''
