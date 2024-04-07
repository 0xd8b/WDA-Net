import cv2
import numpy as np


def standar_gaussian(kernel_size):
    gt = np.zeros((kernel_size, kernel_size))
    gt[int(kernel_size/2)][int(kernel_size/2)] = 1
    count_map = cv2.GaussianBlur(gt, (kernel_size, kernel_size), 0, borderType=0)
    am = np.amax(count_map)
    return 255/am

if __name__ ==  "__main__":

    cofficient = standar_gaussian(61)