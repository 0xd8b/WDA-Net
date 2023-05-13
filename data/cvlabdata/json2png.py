import os
import json
import numpy as np
import cv2
def save_img(filename,img):
    cv2.imwrite(filename, img)

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def generate_sparse_label(binary_map, pnt, save_path):

    _, connections = cv2.connectedComponents(binary_map)
    unique = np.unique(connections * pnt)[1:]

    pred_sparse_map = np.in1d(connections, unique).reshape(binary_map.shape).astype('uint8')
    save_img(save_path, pred_sparse_map*255)


# 文件夹路径
lab_dir = r'./train/lab'

pnt_save_dir = r"C:\Users\CUDAF\PycharmProjects\WDA-Net\data\cvlabdata\train\sparsepnt_15%"
make_dirs(pnt_save_dir)
plb_save_dir = r"C:\Users\CUDAF\PycharmProjects\WDA-Net\data\cvlabdata\train\sparselab_15%"
make_dirs(plb_save_dir)
# 读取包含非零位置坐标的json文件
with open("./train/cvlab_30%.json", "r") as f:
    data = json.load(f)

# 遍历文件夹中的所有.png图像，生成对应的标记图像
for item in data[1]:
    for key, value in item.items():
        # 读取图像
        pnt = np.zeros((int(data[0][0]['width']),int(data[0][1]['height'])))
        lab = cv2.imread(os.path.join(lab_dir, key), -1)
        # 遍历坐标值，将像素值设为1
        for coord in value:
            pnt[coord[0], coord[1]] = 1

        generate_sparse_label(lab, pnt, os.path.join(plb_save_dir, key))
        # 生成保存标记后的图像的文件名
        save_path = os.path.join(pnt_save_dir, key)

        # 将标记后的图像保存到文件夹中
        cv2.imwrite(save_path, pnt)

# 输出结果
print("Saved pnt images to folder")
