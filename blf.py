import sys

import numpy as np
import cv2
import math
from multiprocessing import Pool

FILTER_DIAMETER = 5
SIGMA_I = 75
SIGMA_S = 75
PROC_NUM = 4


def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)


def gaussian(x, sigma):
    return (1 / (2 * math.pi * (sigma ** 2))) * math.exp(-(x ** 2) / (2 * (sigma ** 2)))


def apply_bilateral_filter(source, filtered_image, row, col, diameter, sigma_r, sigma_s):
    hl = int(diameter/2)
    i_filtered = 0
    wp = 0
    i = 0
    while i < diameter:
        j = 0
        neighbour_row = int(row - (hl - i))
        if 0 <= neighbour_row < len(source):
            while j < diameter:
                neighbour_col = int(col - (hl - j))
                if 0 <= neighbour_col < len(source):
                    gr = gaussian(abs(source[row][col] - source[neighbour_row][neighbour_col]), sigma_r)
                    gs = gaussian(distance(row, col, neighbour_row, neighbour_col), sigma_s)
                    print("************")
                    print("the point i want to color: " "( " + str(row) + "," + str(col) + ")")
                    print("the neighbor: (" + str(neighbour_row) + ", " + str(neighbour_col) + ")")
                    print("gr: " + str(gr))
                    print("gs: " + str(gs))
                    # w = gr * gs
                    w = gr
                    i_filtered += w * source[neighbour_row][neighbour_col]
                    wp += w
                j += 1
        i += 1
    i_filtered = i_filtered / wp
    filtered_image[row][col] = int(i_filtered)
    print("the color val of the new pixle is:!")
    print(i_filtered)


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

    # cv2.imwrite("pixelim.jpg", pixels)
    # arr = np.array(pixels, dtype=np.uint8)
    # data = np.zeros((3, 3, 3), dtype=np.uint8)
    data = np.zeros((3, 3, 1), dtype=np.uint8)
    data[0, 0] = 255
    data[0, 1] = 255
    data[0, 2] = 255
    data[1, 0] = 255
    data[1, 1] = 0
    data[1, 2] = 255
    data[2, 0] = 255
    data[2, 1] = 255
    data[2, 2] = 255

    # data[0, 0] = [50, 50, 50]
    # data[0, 1] = [100, 100, 100]
    # data[0, 2] = [150, 150, 150]
    # data[1, 0] = [255, 255, 255]
    # data[1, 1] = [0, 0, 0]
    # data[1, 2] = [255, 255, 255]
    # data[2, 0] = [150, 150, 150]
    # data[2, 1] = [100, 100, 100]
    # data[2, 2] = [50, 50, 50]

    cv2.imwrite("amypix.png", data)
    # kid = cv2.imread(str(sys.argv[1]), 0)
    # blue, green, red = cv2.split(data)
    pool = Pool(processes=PROC_NUM)
    # future_blue, future_green, future_red = pool.map(bilateral_filter_own, [(blue, FILTER_DIAMETER, SIGMA_I, SIGMA_S),
    #                                                                         (green, FILTER_DIAMETER, SIGMA_I, SIGMA_S),
    #                                                                         (red, FILTER_DIAMETER, SIGMA_I, SIGMA_S)])
    grey_kid_filtered = bilateral_filter_own(args=[data, FILTER_DIAMETER, SIGMA_I, SIGMA_S])
    filter_opencv = cv2.bilateralFilter(data, FILTER_DIAMETER, SIGMA_I, SIGMA_S)
    cv2.imwrite("how should look.jpg", filter_opencv)
    # mered_own = cv2.merge((future_blue, future_green, future_red))
    # cv2.imwrite("my merge.jpg", mered_own)
    cv2.imwrite("my merge.jpg", grey_kid_filtered)
