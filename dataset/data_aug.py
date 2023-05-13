import cv2
import numpy as np
import random


def center_cropping(image, y, x, dim1, dim2):
    cropped_img = image[dim1 - y // 2:dim1 + y // 2, dim2 - x // 2:dim2 + x // 2]

    return cropped_img

def cut_aug(img_A, pnt_A, img_B, pnt_B):

    cut_size = [256, 256]
    kernel = np.ones((cut_size[0], cut_size[1]), np.uint8)
    density_map = cv2.filter2D(pnt_B, -1, kernel, borderType=0)

    density_map[:cut_size[0] // 2, :] = float("-inf")
    density_map[-cut_size[0] // 2:, :] = float("-inf")
    density_map[:, :cut_size[1] // 2] = float("-inf")
    density_map[:, -cut_size[1] // 2:] = float("-inf")

    row, col = np.where(density_map == np.max(density_map))

    center_index = random.randint(0, row.size - 1)
    center_row = row[center_index]
    center_col = col[center_index]

    img_B_cut = center_cropping(img_B, cut_size[0], cut_size[1], center_row, center_col)
    pnt_B_cut = center_cropping(pnt_B, cut_size[0], cut_size[1], center_row, center_col)

    density_map = cv2.filter2D(pnt_A, -1, kernel, borderType=0)
    density_map[:cut_size[0] // 2, :] = float("inf")
    density_map[-cut_size[0] // 2:, :] = float("inf")
    density_map[:, :cut_size[1] // 2] = float("inf")
    density_map[:, -cut_size[1] // 2:] = float("inf")

    row, col = np.where(density_map == np.min(density_map))

    center_index = random.randint(0, row.size - 1)
    center_row = row[center_index]
    center_col = col[center_index]

    img_A[center_row - cut_size[0] // 2:center_row + cut_size[0] // 2,
    center_col - cut_size[1] // 2:center_col + cut_size[1] // 2] = img_B_cut
    pnt_A[center_row - cut_size[0] // 2:center_row + cut_size[0] // 2,
    center_col - cut_size[1] // 2:center_col + cut_size[1] // 2] = pnt_B_cut

    return  img_A, pnt_A