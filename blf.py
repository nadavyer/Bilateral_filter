import numpy as np
import cv2
import sys
import math
from multiprocessing import Pool

FILTER_DIAMETER = 5
SIGMA_I = 75
SIGMA_S = 75
PROC_NUM = 4


def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter / 2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)

            if neighbour_x < 0:
                neighbour_x = -neighbour_x
            elif neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y < 0:
                neighbour_y = -neighbour_y
            elif neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[int(neighbour_x)][int(neighbour_y)] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[int(neighbour_x)][int(neighbour_y)] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = int(round(i_filtered))


def bilateral_filter_own(args):
    source, filter_diameter, sigma_i, sigma_s = args
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image


if __name__ == "__main__":
    rb256 = cv2.imread(str(sys.argv[1]))
    blue, green, red = cv2.split(rb256)
    cv2.imwrite("blue.jpg", blue)
    cv2.imwrite("green.jpg", green)
    cv2.imwrite("red.jpg", red)
    merged = cv2.merge((blue, green, red))
    cv2.imwrite("merged.jpg", merged)
    pool = Pool(processes=PROC_NUM)
    future_blue, future_green, future_red = pool.map(bilateral_filter_own, [(blue, FILTER_DIAMETER, SIGMA_I, SIGMA_S),
                                                                            (green, FILTER_DIAMETER, SIGMA_I, SIGMA_S),
                                                                            (red, FILTER_DIAMETER, SIGMA_I, SIGMA_S)])

    filter_opencv = cv2.bilateralFilter(rb256, 5, 75, 75)
    cv2.imwrite("how should look.jpg", filter_opencv)
    # filtered_blue = bilateral_filter_own(blue, 5, 75, 75)
    # filtered_green = bilateral_filter_own(green, 5, 75, 75)
    # filtered_red = bilateral_filter_own(red, 5, 75, 75)
    mered_own = cv2.merge((future_blue, future_green, future_red))
    cv2.imwrite("my merge.jpg", mered_own)

    # myselfimg = cv2.imread(str(sys.argv[2]), 0)
    # filter_opencv = cv2.bilateralFilter(x32, 15, 300.0, 300.0)
    # our_filter = bilateral_filter_own(x32, 9, 100.0, 100.0)
    # blury = bilateral_filter_own(myselfimg, 9, 150.0, 150.0)
    # cv2.imwrite("our_filter.png", our_filter)
    # cv2.imwrite("opencv.png", filter_opencv)
