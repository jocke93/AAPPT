import math
import os

import numpy as np
from cv2 import cv2


def read_border_templates(path=fr'.{os.sep}borders'):
    images_paths = [f'{path}{os.sep}TL.png', f'{path}{os.sep}TR.png', f'{path}{os.sep}BR.png', f'{path}{os.sep}BL.png']
    borders = dict()
    for path in images_paths:
        key = path[path.rfind(r'.') - 2:path.rfind(r'.')]
        borders[key] = cv2.imread(path)
        borders[key] = cv2.cvtColor(borders[key], cv2.COLOR_BGR2GRAY)
        borders[key] = cv2.threshold(borders[key], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return borders


def show_image(name, image, factor, wait=False):
    h, w = image.shape[:2]
    scaled = cv2.resize(image, (int(w * factor), int(h * factor)))
    cv2.imshow(name, scaled)
    if wait:
        cv2.waitKey(0)


def resize_image(image, percent=20):
    dim = int(image.shape[1] * percent / 100), int(image.shape[0] * percent / 100)
    return cv2.resize(image, dim)


def scale_image(image, factor):
    h, w = image.shape[:2]
    return cv2.resize(image, (math.ceil(w * factor), math.ceil(h * factor)))


# positive angle is counterclockwise, negative is clockwise
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=[255, 255, 255])
    return result


def morph_image(image, test_logger):
    test_logger.info('Morphing image...')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    (_, gray_inv) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    (T, thresh_inv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    morph = thresh_inv.copy()
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    horizontal_lines = cv2.morphologyEx(morph, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
    vertical_lines = cv2.morphologyEx(morph, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    image_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # thresh_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, ellipse_kernel, iterations=1)
    # return gray_inv, thresh_inv, thresh_inv
    test_logger.info('Morphing image DONE.')
    return gray_inv, thresh_inv, image_lines
