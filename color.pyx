import cython
cimport numpy
import cv2 as cv
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
#def colorfull_fast(numpy.ndarray[numpy.uint8_t, ndim=3, mode="c"] frame):
cpdef numpy.ndarray[numpy.uint8_t, ndim=3, mode="c"] colorfull_fast(numpy.ndarray[numpy.int64_t, ndim=2, mode="c"] frame, numpy.ndarray[numpy.uint8_t, ndim=3, mode="c"] out, int width, int height):
  # set the variable extension types
  cdef int x, y, b, g, r

  #frame = cv.imdecode(np.frombuffer(byteframe, np.uint8), -1)

  # loop over the image, pixel by pixel
  for x in range(width):
    for y in range(height):
      r = frame[x, y]
      if r == 0: #background
        out[x, y] = (0,0,0)
      elif r == 1: #roadAsphalt
        out[x, y] = (85,85,255)
      elif r == 2: #roadPaved
        out[x, y] = (85,170,127)
      elif r == 3: #roadUnpaved
        out[x, y] = (255,170,127) 
      elif r == 4: #roadMarking
        out[x, y] = (255,255,255) 
      elif r == 5: #speedBump
        out[x, y] = (255,85,255)
      elif r == 6: #catsEye
        out[x, y] = (255,255,127)          
      elif r == 7: #stormDrain
        out[x, y] = (170,0,127) 
      elif r == 8: #manholeCover
        out[x, y] = (0,255,255) 
      elif r == 9: #patchs
        out[x, y] = (0,0,127) 
      elif r == 10: #waterPuddle
        out[x, y] = (170,0,0)
      elif r == 11: #pothole
        out[x, y] = (255,0,0)
      elif r == 12: #cracks
        out[x, y] = (255,85,0)
 
  out = cv.cvtColor(out,cv.COLOR_BGR2RGB)
  return out
  # return the colored image


@cython.boundscheck(False)
@cython.wraparound(False)
#def colorfull_fast(numpy.ndarray[numpy.uint8_t, ndim=3, mode="c"] frame):
cpdef numpy.ndarray[numpy.uint8_t, ndim=2, mode="c"] color_roads(numpy.ndarray[numpy.uint8_t, ndim=2, mode="c"] frame):
  # set the variable extension types
  cdef int x, y, width, height, b, g, r

  #frame = cv.imdecode(np.frombuffer(byteframe, np.uint8), -1)

  # grab the image dimensions
  width = frame.shape[0]
  height = frame.shape[1]
    
  # loop over the image, pixel by pixel
  for x in range(width):
    for y in range(height):
      r = frame[x, y]
      if r == 1: #roadAsphalt
        frame[x, y] = 255
      elif r == 2: #roadPaved
        frame[x, y] = 255
      elif r == 3: #roadUnpaved
        frame[x, y] = 255
      elif r == 4: #roadMarking
        frame[x, y] = 255
      else:
        frame[x, y] = 0
      
  # return the colored image
  return frame