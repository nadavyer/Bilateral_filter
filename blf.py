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
    hl = int(diameter/2)
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        neighbour_x = int(x - (hl - i))
        if neighbour_x >= len(source) or neighbour_x < 0:
            i += 1
            continue
        while j < diameter:
            neighbour_y = int(y - (hl - j))
            if neighbour_y >= len(source) or neighbour_y < 0:
                j += 1
                continue
            gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = i_filtered


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
    merged = cv2.merge((blue, green, red))
    cv2.imwrite("merged.jpg", merged)
    pool = Pool(processes=PROC_NUM)
    future_blue, future_green, future_red = pool.map(bilateral_filter_own, [(blue, FILTER_DIAMETER, SIGMA_I, SIGMA_S),
                                                                            (green, FILTER_DIAMETER, SIGMA_I, SIGMA_S),
                                                                            (red, FILTER_DIAMETER, SIGMA_I, SIGMA_S)])
    filter_opencv = cv2.bilateralFilter(rb256, FILTER_DIAMETER, SIGMA_I, SIGMA_S)
    cv2.imwrite("how should look.jpg", filter_opencv)
    mered_own = cv2.merge((future_blue, future_green, future_red))
    cv2.imwrite("my merge.jpg", mered_own)
