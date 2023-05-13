import numpy as np
import os
import cv2
import math
from skimage import morphology
import random
import json

def generate_sparse_label(binary_map, pnt, save_path):

    _, connections = cv2.connectedComponents(binary_map)
    unique = np.unique(connections * pnt)[1:]

    pred_sparse_map = np.in1d(connections, unique).reshape(binary_map.shape).astype('uint8')
    save_img(save_path, pred_sparse_map*255)

def generate_certain_proportion_point(binary_map, ratio):

    binary_map = binary_map.astype('uint8')
    connections_num, connections = cv2.connectedComponents(binary_map)
    dist_img = cv2.distanceTransform(binary_map, cv2.DIST_L2, 3)

    gt_point = np.zeros(dist_img.shape[:2])

    select = random.sample(range(1, connections_num), math.ceil(ratio*connections_num))
    for j in select:
        j_c = connections.copy()
        j_c[j_c != j] = 0
        dist = dist_img * j_c
        max_index = np.unravel_index(np.argmax(dist), dist.shape)
        gt_point[max_index[:]] = 1

    return  gt_point

def pointmap_to_gaussianmap(pointmap, sigma):

    pointmap[pointmap !=0] = 1
    pointmap = pointmap.astype('float32')
    gaussianmap = cv2.GaussianBlur(pointmap, sigma, 0 ,borderType=0)
    am = np.amax(gaussianmap)
    if am != 0:
        gaussianmap /= am / 255
    else:
        gaussianmap = gaussianmap

    return gaussianmap

def save_img(filename,img):
    cv2.imwrite(filename, img)

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def numpy_to_builtin(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, np.float64):
        return float(data)
    else:
        return data


pnt_dir = r''
setup_seed(20)
result = []
wh_result = []
flag = 1
for file_name in sorted(os.listdir(pnt_dir)):
    print(file_name)
    pnt_path = os.path.join(pnt_dir, file_name)
    pnt = cv2.imread(pnt_path, -1)

    if flag == 1:
        flag = 0
        wh_result.append([{'width': pnt.shape[0]}, {'height': pnt.shape[1]}])

    non_zero_pixels = np.nonzero(pnt)
    # 将结果添加到列表中
    result.append({file_name: list(map(numpy_to_builtin, zip(non_zero_pixels[0], non_zero_pixels[1])))})

wh_result.append(result)
# 保存结果到json文件中
with open("train/cvlab_15%_split3.json", "w") as f:
    json.dump(wh_result, f, default=numpy_to_builtin)




