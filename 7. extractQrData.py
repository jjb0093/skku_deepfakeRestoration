import jpegio as jio
import numpy as np
import cv2

def extractDct(point, coef):
    hc, wc = coef.shape
    hy, hx = hc//8, wc//8

    dct = np.zeros((hy, hx), dtype = np.int32)

    for i in range(hy):
        for k in range(hx):
            dct[i, k] = coef[i*8+point[0], k*8+point[1]] & 1

    return dct

def dctImage(dct):
    dct = dct * 255
    image = cv2.resize(dct, (dct.shape[1]*10, dct.shape[0]*10), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite("dctImg.png", image)

path = "testImage_stegano.jpg"

img = jio.read(path)
coef = img.coef_arrays[0]

point = (1, 2)
dct = extractDct(point, coef)
dctImage(dct)