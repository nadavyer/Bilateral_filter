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


def apply_bilateral_filter(source, filtered_image, row, col, diameter, sigma_i, sigma_s):
    hl = int(diameter/2)
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        neighbour_row = int(row - (hl - i))
        if len(source) > neighbour_row >= 0:
            while j < diameter:
                neighbour_col = int(col - (hl - j))
                if len(source) > neighbour_col >= 0:
                    gi = gaussian(source[neighbour_row][neighbour_col] - source[row][col], sigma_i)
                    gs = gaussian(distance(neighbour_row, neighbour_col, row, col), sigma_s)
                    w = gi * gs
                    i_filtered += source[neighbour_row][neighbour_col] * w
                    Wp += w
                j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[row][col] = i_filtered


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

    pixels = [[(10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120),  (130, 140, 150), (200, 200, 200)]]
    # cv2.imwrite("pixelim.jpg", pixels)
    # arr = np.array(pixels, dtype=np.uint8)
    rb256 = cv2.imread(str(sys.argv[1]))
    blue, green, red = cv2.split(rb256)
    pool = Pool(processes=PROC_NUM)
    future_blue, future_green, future_red = pool.map(bilateral_filter_own, [(blue, FILTER_DIAMETER, SIGMA_I, SIGMA_S),
                                                                            (green, FILTER_DIAMETER, SIGMA_I, SIGMA_S),
                                                                            (red, FILTER_DIAMETER, SIGMA_I, SIGMA_S)])
    filter_opencv = cv2.bilateralFilter(rb256, FILTER_DIAMETER, SIGMA_I, SIGMA_S)
    cv2.imwrite("how should look.jpg", filter_opencv)
    mered_own = cv2.merge((future_blue, future_green, future_red))
    cv2.imwrite("my merge.jpg", mered_own)
