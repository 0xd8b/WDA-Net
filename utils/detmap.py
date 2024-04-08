import cv2
import numpy as np
from skimage import measure
from skimage import morphology as morph

def make_gaussian_map(point_map, sigma):

    count_map = cv2.GaussianBlur(point_map.astype('float32'), sigma, 0, borderType=0)
    am = np.amax(count_map)
    if am != 0:
        count_map /=am/255.0
    else:
        count_map = count_map

    return count_map

def generate_gaussianmap(binary_map, sigma):

    binary_map = binary_map.astype('uint8')
    retval, connections = cv2.connectedComponents(binary_map)
    dist_img = cv2.distanceTransform(binary_map, cv2.DIST_L2, 3)

    gt = np.zeros(dist_img.shape[:2])
    for j in range(1, retval):
        j_c = connections.copy()
        j_c[j_c != j] = 0
        dist = dist_img * j_c
        max_index = np.unravel_index(np.argmax(dist), dist.shape)
        gt[max_index[:]] = 1

    det_map = cv2.GaussianBlur(gt, sigma, 0, borderType=0)
    if np.count_nonzero(gt)!=0:
        am = np.min(det_map[gt>0])
        if am != 0:
            det_map /= am
            det_map[det_map > 1] = 1
            det_map = det_map*255.0
        else:
            det_map = det_map

    return gt, det_map

def getcenter_makegaussian(binary_map, sigma):

    binary_map = binary_map.astype('uint8')

    labels = measure.label(binary_map, connectivity=1)

    properties = measure.regionprops(labels)

    gt = np.zeros_like(binary_map).astype('float')
    for prop in properties:
        # print(int(prop.centroid[0]), int(prop.centroid[1]))
        gt[int(prop.centroid[0]), int(prop.centroid[1])] = 1

    count_map = cv2.GaussianBlur(gt, sigma, 0, borderType=0)
    am = np.amax(count_map)
    if am != 0:
        count_map /= am/255.0
    else:
        count_map = count_map

    return gt, count_map

def get_blobs(probs):
    pred_mask = probs.astype('uint8')
    blobs = morph.label(pred_mask == 1)

    return blobs
