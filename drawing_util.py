from cv2 import cv2


def draw_contours(image, contours, color, thickness):
    for cont in contours:
        cv2.drawContours(image, [cont], -1, color=color, thickness=thickness)


def draw_rectangles(image, rectangles, color, thickness):
    for r in rectangles:
        cv2.rectangle(image, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), color=color, thickness=thickness)


def draw_circles(image, circles, color, thickness):
    for c in circles:
        cv2.circle(image, (int(c[0][0]), int(c[0][1])), int(c[1]), color=color, thickness=thickness)
