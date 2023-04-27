from cv2 import cv2
import math
import numpy as np


def is_square_contour(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    return len(approx) == 4


def is_circle_contour(contour, tolerance=0.85):
    peri = cv2.arcLength(contour, True)
    if not peri:
        return False
    circ = 4 * math.pi * cv2.contourArea(contour) / (peri * peri)
    return circ > tolerance


def same_row_rect(r1, r2, tolerance=0.25):
    return r2[1] <= r1[1] <= r2[1] + tolerance * r2[3] or r1[1] <= r2[1] <= r1[1] + tolerance * r1[3]


def same_col_rect(r1, r2, tolerance=0.25):
    return r2[0] <= r1[0] <= r2[0] + tolerance * r2[2] or r1[0] <= r2[0] <= r1[0] + tolerance * r1[2]


def same_row_circle(c1, c2, r_med):
    return c2[0][1] - r_med <= c1[0][1] <= c2[0][1] + r_med and \
           c1[0][1] - r_med <= c2[0][1] <= c1[0][1] + r_med


def same_col_circle(c1, c2, r_med):
    return c2[0][0] - r_med <= c1[0][0] <= c2[0][0] + r_med and \
           c1[0][0] - r_med <= c2[0][0] <= c1[0][0] + r_med


def get_grid_cells(rectangles):
    grid = []
    try:
        for r in rectangles:
            flag = False
            for row in grid:
                if len(row) and same_row_rect(r, row[-1]):
                    flag = True
                    row.append(r)
                    break
            if not flag:
                grid.append([r])

        for row in grid:
            row.sort(key=lambda el: el[0])  # el[0] is x coordinate
        grid.sort(key=lambda el: el[0][1])  # el[0] is a rectangle, el[0][1] is y coordinate

        cnt = None
        for i, row in enumerate(grid):
            if cnt is None:
                cnt = len(row)
            elif len(row) != cnt:
                raise Exception(f'Row {i} has {len(row)} cells, initial had {cnt} cells')

        if cnt:
            for j in range(len(grid[0])):
                for i in range(len(grid) - 1):
                    if not same_col_rect(grid[i][j], grid[i + 1][j]):
                        raise Exception(f'Different column {i, j}: {grid[i][j]} and {i + 1, j}: {grid[i + 1][j]}')

    except Exception as e:
        raise e

    return grid


def get_grid_circles(circles):
    grid = []
    try:
        r_med = int(np.median([c[1] for c in circles]))
        for c in circles:
            flag = False
            for row in grid:
                if len(row) and same_row_circle(c, row[-1], r_med):
                    flag = True
                    row.append(c)
                    break
            if not flag:
                grid.append([c])

        for row in grid:
            row.sort(key=lambda el: el[0])  # el[0] is x coordinate
        grid.sort(key=lambda el: el[0][0][1])  # el[0] is circle, el[0][0][1] is y coordinate

        cnt = None
        for i, row in enumerate(grid):
            if cnt is None:
                cnt = len(row)
            elif len(row) != cnt:
                raise Exception(f'Row {i} has {len(row)} circles, initial had {cnt} circles')

        if cnt:
            for j in range(len(grid[0])):
                for i in range(len(grid) - 1):
                    if not same_col_circle(grid[i][j], grid[i + 1][j], r_med):
                        raise Exception(f'Different column {i, j}: {grid[i][j]} and {i + 1, j}: {grid[i + 1][j]}')

    except Exception as e:
        raise e

    return grid


def count_circle_filled_pixels(image, radius_factor):
    cen = [image.shape[1] // 2, image.shape[0] // 2]
    r = min(image.shape[1], image.shape[0]) // 2 * radius_factor
    result = 0
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            result += 1 if math.sqrt(math.pow(cen[0] - i, 2) + math.pow(cen[1] - j, 2)) < r and image[j][i] else 0
    return result


def is_rect_inside_another(rect, another):
    return (another[0] < rect[0] < another[0] + another[2]) and \
           (another[0] < rect[0] + rect[2] < another[0] + another[2]) and \
           (another[1] < rect[1] < another[1] + another[3]) and \
           (another[1] < rect[1] + rect[3] < another[1] + another[3])


def rect_overlapping_perc(rect1, rect2):
    surf_inter = max(0, min(rect1[0] + rect1[2], rect2[0] + rect2[2]) - max(rect1[0], rect2[0])) * \
                 max(0, min(rect1[1] + rect1[3], rect2[1] + rect2[3]) - max(rect1[1], rect2[1]))
    surf1 = rect1[2] * rect1[3]
    # surf2 = rect2[2] * rect2[3]
    # return surf_inter / (surf1 + surf2 - surf_inter)
    return surf_inter / surf1
