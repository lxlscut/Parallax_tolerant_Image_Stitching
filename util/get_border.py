import cv2
import numpy as np


def get_border(img, method):
    img = img/255.
    B = np.ones([3, 3])
    C = cv2.filter2D(src=img, kernel=B, ddepth=-1, delta=0, borderType=cv2.BORDER_DEFAULT)
    if method == "inside":
        result = C < 9 & img
        return result
    elif method == "outside":
        result = C > 0 & (~img)
        return result
    else:
        raise Exception("please choose the right method")
