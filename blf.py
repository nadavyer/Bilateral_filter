import numpy as np
import cv2
import sys
import math


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter/2
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


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
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
    textimg = cv2.imread(str(sys.argv[1]), 0)
    myselfimg = cv2.imread(str(sys.argv[2]), 0)
    filter_opencv = cv2.bilateralFilter(myselfimg, 9, 100.0, 100.0)
    filtered_image_own = bilateral_filter_own(myselfimg, 9, 100.0, 100.0)
    # blury = bilateral_filter_own(myselfimg, 9, 150.0, 150.0)
    cv2.imwrite("us.png", filtered_image_own)
    cv2.imwrite("opencv.png", filter_opencv)