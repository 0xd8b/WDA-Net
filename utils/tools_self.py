import SimpleITK as sitk
import numpy as np
import nibabel, math
import sys
import torch
import random
import datetime
import os
import cv2
import tqdm
import os
import torch
import matplotlib.pyplot as plt
import torchvision
import shutil
from skimage import morphology
import torch.nn.functional as F

offset=1
##########
def center_cropping(image, y, x, dim1, dim2):
    cropped_img = image[dim1 - y // 2:dim1 + y // 2, dim2 - x // 2:dim2 + x // 2]

    return cropped_img

def mask_background(partial_label_dir, pseudo_label_dir, save_path):
    make_dirs(save_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(partial_label_dir))):
        partial_label_path = os.path.join(partial_label_dir, file_name)
        pseudo_label_path = os.path.join(pseudo_label_dir, file_name)
        partial_pseudo_label_save_path = os.path.join(save_path, file_name)

        partial_label = cv2.imread(partial_label_path, -1)
        pseudo_label = cv2.imread(pseudo_label_path, -1)

        partial_label[partial_label != 0] = 1
        partial_label[partial_label == 0] = 255
        partial_label[pseudo_label == 0] = 0

        save_img(partial_pseudo_label_save_path, partial_label)

def generate_slabel(sparse_point_map, predict_map):

    sparse_point_map[sparse_point_map>0]=255

    sparse_point_map[sparse_point_map > 0] = 1
    _, connections = cv2.connectedComponents(predict_map)
    unique = np.unique(connections * sparse_point_map)[1:]

    pred_sparse_map = np.in1d(connections, unique).reshape(predict_map.shape).astype('uint8')

    return  pred_sparse_map*255

def generate_slabel_1(A, B):

    A[A > 0] = 1
    _, connections = cv2.connectedComponents(B)
    unique = np.unique(connections * A)[1:]

    C = np.in1d(connections, unique).reshape(B.shape).astype('uint8')

    return  C
##########
def generate_sparse_plabel(sparse_point_path, predictlabel_path, save_path):

    make_dirs(save_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(sparse_point_path))):

        plabel_path = os.path.join(predictlabel_path, file_name)
        pointmap_path = os.path.join(sparse_point_path, file_name)

        predict_map = cv2.imread(plabel_path, -1).astype('uint8')
        sparse_point_map = cv2.imread(pointmap_path, -1).astype('uint8')

        if len(np.unique(predict_map)) >= 3:
            predict_map[predict_map != 1] = 0

        sparse_point_map[sparse_point_map > 0] = 1
        _, connections = cv2.connectedComponents(predict_map)
        unique = np.unique(connections * sparse_point_map)[1:]

        pred_sparse_map = np.in1d(connections, unique).reshape(predict_map.shape).astype('uint8')

        kernel = np.ones((3, 3), np.uint8)
        point_dilate = cv2.dilate(sparse_point_map, kernel, iterations=4)

        add_map = cv2.bitwise_or(pred_sparse_map, point_dilate)

        save_filename = os.path.join(save_path, file_name)

        save_img(save_filename, add_map*255)
##########
def min_max(image, max, min):
    image_new = (image - np.min(image)) * (max - min) / ((np.max(image) - np.min(image)) + min + 1e-6)
    return image_new

def conver_tensor_to_numpy(tensordata):
    numpydata = tensordata.cpu().numpy()
    return numpydata

def check(a , row, col):
    pad = np.pad(a,((offset,offset),(offset,offset)),'constant', constant_values=(0,0))
    row_offset = row+offset
    col_offset = col+offset
    sum = np.sum(pad[row_offset-offset:row_offset+offset, col_offset-offset:col_offset+offset])
    if sum > 1:
        a[row][col] = 0
    return a

class Logger(object):
    def __init__(self, filename='logprocess.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    # print(worker_id)
    GLOBAL_WORKER_ID = worker_id
    setup_seed(worker_id + 20)

def compute_entropy(arr, num_classes):
    tensor = torch.from_numpy(arr)
    predicted_entropy = torch.sum(torch.mul(tensor,torch.log(tensor)), dim=2) * (-1/np.log(num_classes))
    return predicted_entropy.numpy()


def display_logger(learning_rate, i_iter, num_steps, loss_seg_value, sdice, sjac):

    print('time = {0},lr = {1: 5f}'.format(datetime.datetime.now(), learning_rate))

    print('iter = {0:8d}/{1:8d}, loss_seg = {2:.5f}'.format(i_iter, num_steps, loss_seg_value))

    print('sdice2 = {0:.5f} sjac2 = {1:.5f}'.format(sdice, sjac,))

##########
def makedatalist(imgpath, listpath):
    num = 0
    file_path = imgpath
    path_list = os.listdir(file_path)
    for file_name in path_list:
        path = imgpath + file_name+'/'
        if os.path.isdir(path):
            num = num + 1
            path_list.extend(os.listdir(path))
        else:
            break

    del path_list[:num]
    path_list.sort()

    with open(listpath, 'a') as f:
        f.seek(0)
        f.truncate()
        for file_name in path_list:
            f.write(file_name + '\n')

    f.close()

def cnn_paras_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters')
    return total_params, total_trainable_params

def save_img(filename,img):
    cv2.imwrite(filename, img)

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        # print('create dir:',dir)

def save_train_img_for_debug(image_tensor, iter=0, save_dir=None):
    # if iter%100:
    image_batch = torchvision.utils.make_grid(image_tensor, padding=2)
    image_numpy = image_batch.cpu().numpy()
    plt.imshow(np.transpose(image_numpy, (1,2,0)))
    plt.show()
##########
def remove_or_create_exp_dir(exp_name):
    print(exp_name)
    if os.path.exists(exp_name):
        wait = input("do you want to rm the dir,'y'or'n'\n")
        if wait == 'y':
            print('rm the exp dir\n')
            shutil.rmtree(exp_name)
        else:
            make_dirs(exp_name)

def bak_code(code_path_list,exp_name):
    for path in code_path_list:
        make_dirs(exp_name + '/code_bak/')
        shutil.copyfile(path, exp_name + '/code_bak/'+path.split('/')[-1])
        print('bak change code in the path:',exp_name + '/code_bak/'+path.split('/')[-1])
        # shutil.copyfile(path, exp_name + '/code_bak/'+path.split('/')[-1])
    # shutil.copytree('dataset', exp_name + '/code_bak/dataset')
    # shutil.copytree('model', exp_name + '/code_bak/model')
    # shutil.copytree('utils', exp_name + '/code_bak/utils')
    # shutil.copyfile('Step1_Innitialy.py', exp_name + '/code_bak/Step1_Innitialy.py')
    # shutil.copyfile('add_arguments.py', exp_name + '/code_bak/add_arguments.py')

def peak_local_maxima(input, win_size, ratio):

    input = np.expand_dims(input, axis=0)
    input = np.expand_dims(input, axis=0)
    input_as_tensor = torch.from_numpy(input.astype('float32')).float().cuda()
    input_as_tensor[input_as_tensor < torch.max(input_as_tensor * 0.5)] = 0
    offset = (win_size - 1) // 2
    padding = torch.nn.ConstantPad2d(offset,float("-inf"))
    padded_maps = padding(input_as_tensor)
    batch_size, num_channels,h,w = padded_maps.size()
    element_map = torch.arange(0,h*w).long().view(1,1,h,w)[:,:,offset:-offset,offset:-offset].float().cuda()

    _, indices = F.max_pool2d(padded_maps, kernel_size = win_size,stride=1,return_indices = True)
    peak_map = (indices == element_map)

    peak = (peak_map.data[0, 0, :] + 0).cpu().numpy()
    num = np.count_nonzero(peak)

    crm = input_as_tensor * peak_map.float()

    crm1 = input_as_tensor * peak_map.float()
    max_all = []
    element_map1 = torch.arange(0, crm.size()[2] * crm.size()[3]).long().view(1, 1, crm.size()[2], crm.size()[3]).cuda()
    expect_num = np.int(np.ceil(num * ratio))

    for i in range(expect_num):
        max_one, indices = F.max_pool2d(crm1, kernel_size=input_as_tensor.size()[2:], return_indices=True)
        max_all.append(max_one)
        crm1[indices == element_map1] = float("-Inf")

    mask1 = crm1 == float('-Inf')
    mask1 = (mask1 & peak_map)
    mask_array = (mask1.data[0, 0, :] + 0).cpu().numpy()

    return  mask_array


def peak_local_maxima_v1(input, gt_partiallab, win_size, ratio):

    input = np.expand_dims(input, axis=0)
    input = np.expand_dims(input, axis=0)
    input_as_tensor = torch.from_numpy(input.astype('float32')).float().cuda()
    input_as_tensor[input_as_tensor < torch.max(input_as_tensor * 0.5)] = 0
    offset = (win_size - 1) // 2
    padding = torch.nn.ConstantPad2d(offset, float("-inf"))
    padded_maps = padding(input_as_tensor)
    batch_size, num_channels, h, w = padded_maps.size()
    element_map = torch.arange(0,h*w).long().view(1,1,h,w)[:,:,offset:-offset,offset:-offset].float().cuda()

    _, indices = F.max_pool2d(padded_maps, kernel_size = win_size,stride=1,return_indices = True)
    peak_map = (indices == element_map)

    peak = (peak_map.data[0, 0, :] + 0).cpu().numpy()
    num = np.count_nonzero(peak)
    want_num = np.int(np.ceil(num * ratio))
    connect_num, _ = cv2.connectedComponents(gt_partiallab)
    final_num = want_num - connect_num + 1
    peak_filter = peak * gt_partiallab
    peak[peak_filter != 0] = 0

    peak = np.expand_dims(peak, axis=0)
    peak = np.expand_dims(peak, axis=0)
    peak_map = torch.from_numpy(peak>0).cuda()

    # crm = input_as_tensor * peak_map.float()

    crm1 = input_as_tensor * peak_map.float()
    # max_all = []
    # element_map1 = torch.arange(0, crm.size()[2] * crm.size()[3]).long().view(1, 1, crm.size()[2], crm.size()[3]).cuda()
    # expect_num = np.int(np.ceil(num * ratio))

    for i in range(final_num):
        # max_one, indices = F.max_pool2d(crm1, kernel_size=input_as_tensor.size()[2:], return_indices=True)
        # max_all.append(max_one)
        # crm1[indices == element_map1] = float("-Inf")
        indices = torch.argmax(crm1)
        crm1[:, :, indices // crm1.size()[3], indices % (crm1.size()[3])] = float("-Inf")


    mask1 = crm1 == float('-Inf')
    mask1 = (mask1 & peak_map)
    mask_array = (mask1.data[0, 0, :] + 0).cpu().numpy()

    return mask_array

def peak_local_maxima_v2(count, input, gt_partiallab, win_size, ratio):

    input = np.expand_dims(input, axis=0)
    input = np.expand_dims(input, axis=0)
    input_as_tensor = torch.from_numpy(input.astype('float32')).float().cuda()
    # input_as_tensor[input_as_tensor < torch.max(input_as_tensor * 0.5)] = 0
    offset = (win_size - 1) // 2
    padding = torch.nn.ConstantPad2d(offset, float("-inf"))
    padded_maps = padding(input_as_tensor)
    batch_size, num_channels, h, w = padded_maps.size()
    element_map = torch.arange(0,h*w).long().view(1,1,h,w)[:,:,offset:-offset,offset:-offset].float().cuda()

    _, indices = F.max_pool2d(padded_maps, kernel_size = win_size,stride=1,return_indices = True)
    peak_map = (indices == element_map)

    peak = (peak_map.data[0, 0, :] + 0).cpu().numpy()
    # num = np.count_nonzero(peak)
    num = count
    want_num = np.int(np.ceil(num * ratio))
    connect_num, _ = cv2.connectedComponents(gt_partiallab)
    final_num = want_num - connect_num + 1
    peak_filter = peak * gt_partiallab
    peak[peak_filter != 0] = 0

    peak = np.expand_dims(peak, axis=0)
    peak = np.expand_dims(peak, axis=0)
    peak_map = torch.from_numpy(peak>0).cuda()

    crm1 = input_as_tensor * peak_map.float()
    # max_all = []
    # element_map1 = torch.arange(0, crm.size()[2] * crm.size()[3]).long().view(1, 1, crm.size()[2], crm.size()[3]).cuda()
    # expect_num = np.int(np.ceil(num * ratio))

    for i in range(final_num):
        # max_one, indices = F.max_pool2d(crm1, kernel_size=input_as_tensor.size()[2:], return_indices=True)
        # max_all.append(max_one)
        # crm1[indices == element_map1] = float("-Inf")
        indices = torch.argmax(crm1)
        crm1[:, :, indices // crm1.size()[3], indices % (crm1.size()[3])] = float("-Inf")

    mask1 = crm1 == float('-Inf')
    mask1 = (mask1 & peak_map)
    mask_array = (mask1.data[0, 0, :] + 0).cpu().numpy()

    return mask_array

def peak_local_maxima2(input, win_size, ratio):

    input = np.expand_dims(input, axis=0)
    input = np.expand_dims(input, axis=0)
    input_as_tensor = torch.from_numpy(input.astype('float32')).float().cuda()
    input_as_tensor[input_as_tensor < torch.max(input_as_tensor*0.5)] = 0
    offset = (win_size - 1) // 2
    padding = torch.nn.ConstantPad2d(offset,float("-inf"))
    padded_maps = padding(input_as_tensor)
    batch_size, num_channels,h,w = padded_maps.size()
    element_map = torch.arange(0,h*w).long().view(1,1,h,w)[:,:,offset:-offset,offset:-offset].float().cuda()

    _, indices = F.max_pool2d(padded_maps, kernel_size = win_size,stride=1,return_indices = True)
    peak_map = (indices == element_map)

    mask_array = (peak_map.data[0, 0, :] + 0).cpu().numpy()
    num = np.count_nonzero(mask_array)
    nonzero_array = np.nonzero(mask_array)
    index = range(0, len(nonzero_array[0]))
    choice = np.random.choice(index, int(num * ratio), replace=False)
    for i in choice:
        mask_array[nonzero_array[0][i], nonzero_array[1][i]] = 0
    return  mask_array

def from_detectionmap_generate_pseudolab(detectionmap_path, gtpointmap_path, predictmap_path, save_path, save_point_path,ratio):
    make_dirs(save_path)
    make_dirs(save_point_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(detectionmap_path))):
        lab_path = os.path.join(detectionmap_path, file_name)
        pred_lab_path = os.path.join(predictmap_path, file_name)
        gtpointmap_pth = os.path.join(gtpointmap_path, file_name)
        postprocess_labmap_save_path = os.path.join(save_path, file_name)
        postprocess_pointmap_save_path = os.path.join(save_point_path, file_name)

        detection_map = cv2.imread(lab_path, -1)
        predict_map = cv2.imread(pred_lab_path, -1)
        gtpointmap = cv2.imread(gtpointmap_pth, -1)
        gtpointmap[gtpointmap>0]=1

        if len(np.unique(predict_map)) >=3:
            predict_map[predict_map != 1] = 0
        predict_map = morphology.remove_small_objects(predict_map.astype('bool'), min_size=900, connectivity=1)
        predict_map = predict_map.astype('uint8')
        localmaximal_map = peak_local_maxima_v1(detection_map, gtpointmap, win_size=3, ratio=ratio)

        localmaximal_map = cv2.bitwise_or(localmaximal_map.astype('uint8'), gtpointmap)
        final_predict_lab = generate_slabel(localmaximal_map, predict_map)

        save_img(postprocess_labmap_save_path, final_predict_lab)
        save_img(postprocess_pointmap_save_path, localmaximal_map)


def from_detectionmap_generate_pseudolab_v1(detectionmap_path, gtpointmap_path, gtpoint_partiallab_path, predictmap_path, save_path, save_point_path,ratio):
    make_dirs(save_path)
    make_dirs(save_point_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(detectionmap_path))):
        lab_path = os.path.join(detectionmap_path, file_name)
        pred_lab_path = os.path.join(predictmap_path, file_name)
        gtpointmap_pth = os.path.join(gtpointmap_path, file_name)
        gtpoint_partiallab_pth = os.path.join(gtpoint_partiallab_path, file_name)

        postprocess_labmap_save_path = os.path.join(save_path, file_name)
        postprocess_pointmap_save_path = os.path.join(save_point_path, file_name)

        detection_map = cv2.imread(lab_path, -1)
        predict_map = cv2.imread(pred_lab_path, -1)
        gtpointmap = cv2.imread(gtpointmap_pth, -1)
        gtpointmap[gtpointmap > 0] = 1

        gtpoint_partiallab = cv2.imread(gtpoint_partiallab_pth, -1)

        if len(np.unique(predict_map)) >= 3:
            predict_map[predict_map != 1] = 0

        predict_map = morphology.remove_small_objects(predict_map.astype('bool'), min_size=16, connectivity=1)
        predict_map = predict_map.astype('uint8')
        localmaximal_map = peak_local_maxima_v1(detection_map, gtpoint_partiallab, win_size=3, ratio=ratio)

        localmaximal_map = cv2.bitwise_or(localmaximal_map.astype('uint8'), gtpointmap)

        final_predict_lab = generate_slabel(localmaximal_map, predict_map)

        kernel = np.ones((3, 3), np.uint8)
        point_dilate = cv2.dilate(gtpointmap*255, kernel, iterations=4)

        add_map = cv2.bitwise_or(final_predict_lab, point_dilate)

        save_img(postprocess_labmap_save_path, add_map)
        save_img(postprocess_pointmap_save_path, localmaximal_map)


def from_detectionmap_generate_pseudolab_v2(pseudo_count,detectionmap_path, gtpointmap_path, gtpoint_partiallab_path, predictmap_path, save_path, save_point_path,ratio):
    make_dirs(save_path)
    make_dirs(save_point_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(detectionmap_path))):
        lab_path = os.path.join(detectionmap_path, file_name)
        pred_lab_path = os.path.join(predictmap_path, file_name)
        gtpointmap_pth = os.path.join(gtpointmap_path, file_name)
        gtpoint_partiallab_pth = os.path.join(gtpoint_partiallab_path, file_name)

        postprocess_labmap_save_path = os.path.join(save_path, file_name)
        postprocess_pointmap_save_path = os.path.join(save_point_path, file_name)

        detection_map = cv2.imread(lab_path, -1)
        predict_map = cv2.imread(pred_lab_path, -1)
        gtpointmap = cv2.imread(gtpointmap_pth, -1)
        gtpointmap[gtpointmap > 0] = 1

        gtpoint_partiallab = cv2.imread(gtpoint_partiallab_pth, -1)

        if len(np.unique(predict_map)) >= 3:
            predict_map[predict_map != 1] = 0

        predict_map = morphology.remove_small_objects(predict_map.astype('bool'), min_size=16, connectivity=1)
        predict_map = predict_map.astype('uint8')
        # localmaximal_map = peak_local_maxima_v1(detection_map, gtpoint_partiallab, win_size=3, ratio=ratio)
        localmaximal_map = peak_local_maxima_v2(pseudo_count[int(file_name[5:8])-1], detection_map, gtpoint_partiallab, win_size=3, ratio=ratio)

        localmaximal_map = cv2.bitwise_or(localmaximal_map.astype('uint8'), gtpointmap)

        final_predict_lab = generate_slabel(localmaximal_map, predict_map)

        kernel = np.ones((3, 3), np.uint8)
        point_dilate = cv2.dilate(gtpointmap*255, kernel, iterations=4)

        add_map = cv2.bitwise_or(final_predict_lab, point_dilate)

        save_img(postprocess_labmap_save_path, add_map)
        save_img(postprocess_pointmap_save_path, localmaximal_map)


def random_generate_pseudolab(predictmap_path, save_path):
    make_dirs(save_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(predictmap_path))):

        pred_lab_path = os.path.join(predictmap_path, file_name)
        postprocess_labmap_save_path = os.path.join(save_path, file_name)

        predict_map = cv2.imread(pred_lab_path, -1)

        if len(np.unique(predict_map)) >=3:
            predict_map[predict_map != 1] = 0
        predict_map = morphology.remove_small_objects(predict_map.astype('bool'), min_size=900, connectivity=1)
        predict_map = predict_map.astype('uint8')

        retval, connnections = cv2.connectedComponents(predict_map)
        index = range(0, retval-1)
        choice = np.random.choice(index, int(0.5 * (retval-1)), replace=False)
        for i in choice:
            connnections[connnections == i] = 0
        connnections[connnections!=0] = 255
        save_img(postprocess_labmap_save_path, connnections)



def generate_sparse_plabel1(sparse_point_path, predictlabel_path, save_path):

    make_dirs(save_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(sparse_point_path))):

        plabel_path = os.path.join(predictlabel_path, file_name)
        pointmap_path = os.path.join(sparse_point_path, file_name)

        predict_map = cv2.imread(plabel_path, -1).astype('uint8')
        sparse_point_map = cv2.imread(pointmap_path, -1).astype('uint8')

        if len(np.unique(predict_map)) >=3:
            predict_map[predict_map != 1] = 0

        sparse_point_map[sparse_point_map > 0] = 1
        _, connections = cv2.connectedComponents(predict_map)
        unique = np.unique(connections * sparse_point_map)[1:]

        pred_sparse_map = np.in1d(connections, unique).reshape(predict_map.shape).astype('uint8')

        save_filename = os.path.join(save_path, file_name)

        save_img(save_filename, pred_sparse_map*255)

def standar_gaussian(kernel_size):
    gt = np.zeros((kernel_size, kernel_size))
    gt[int(kernel_size/2)][int(kernel_size/2)] = 1
    count_map = cv2.GaussianBlur(gt, (kernel_size, kernel_size), 0, borderType=0)
    am = np.amax(count_map)
    if am != 0:
        count_map /= am / 255
    return 255/am

if __name__ == "__main__":
    pass
