import albumentations as albu
from albumentations import *
from random import randint
import random
import cv2
import PIL.Image as Image
import numpy as np


def strong_aug(p=.5, cropsize=(512, 512)):
    return Compose([
        Flip(),
        Transpose(),
        # RandomGamma(),
        Rotate(),
        RandomBrightnessContrast(),
        MotionBlur(p=0.2),
        # GaussNoise(p=0.2, var_limit=(0, 0.3)),
        # ElasticTransform(p=0.3),
        # Resize(p=0.2, height=cropsize[0], width=cropsize[1]),
        # RandomResizedCrop(height=cropsize[0], width=cropsize[1], always_apply=True),
        # Affine(p=0.5, scale=0.8, translate_percent=0.2, rotate=30, cval=0, shear=30),
        # CropNonEmptyMaskIfExists(height=cropsize[0], width=cropsize[1], always_apply=True),
        RandomCrop(height=cropsize[0], width=cropsize[1], always_apply=True),

    ], p=p)

def strong_aug_exitcrop(p=.5, cropsize=(512, 512)):
    return Compose([
        Flip(),
        Transpose(),
        # RandomGamma(),
        Rotate(),
        RandomBrightnessContrast(),
        MotionBlur(p=0.2),
        # GaussNoise(p=0.2，var_limit=(0, 0.3)),
        ElasticTransform(p=0.3),
        # Affine(p=0.2, scale=0.8, translate_percent=0.2, rotate=30, cval=0, shear=30),
        # CropNonEmptyMaskIfExists(height=cropsize[0], width=cropsize[1], always_apply=True),
        # RandomResizedCrop(height=cropsize[0], width=cropsize[1], scale=(0.25, 1.0), always_apply=True),
        # OneOf([
        #        # Resize(height=cropsize[0], width=cropsize[1], p=0.6),
        #        CropNonEmptyMaskIfExists(height=cropsize[0], width=cropsize[1], p=0.6),
        #        # RandomCrop(height=cropsize[0], width=cropsize[1], p=0.6),
        #        RandomResizedCrop(height=cropsize[0], width=cropsize[1], p=0.4)],
        #        # RandomSizedCrop((256, 512), p=0.4, height=cropsize[0], width=cropsize[1], interpolation=2)],
        #       p=1.0),

        # RandomCrop(height=cropsize[0], width=cropsize[1], always_apply=True),
        # CropNonEmptyMaskIfExists(height=cropsize[0], width=cropsize[1], always_apply=True),
        # Resize(height=cropsize[0], width=cropsize[1], always_apply=True),
        RandomResizedCrop(height=cropsize[0], width=cropsize[1], scale=(0.25, 0.8), always_apply=True),
    ], p=p)


def strong_aug_strong(p=.5, cropsize=(512, 512)):
    return Compose([
        Flip(),
        Transpose(),
        # RandomGamma(),
        Rotate(),
        OneOf([Resize(p=0.2, height=cropsize[0], width=cropsize[1]), RandomSizedCrop((256, 512), p=0.2, height=cropsize[0], width=cropsize[1], interpolation=2)]),
        RandomBrightnessContrast(),
        MotionBlur(),
        ElasticTransform(),
        # ElasticTransform(p=0.3),
    ],p=p)
def count_aug(p=.5, cropsize=(512, 512)):
    return Compose([
        Flip(),
        Transpose(),
        # RandomGamma(),
        Rotate(),
        RandomBrightnessContrast(),
        GaussNoise(p=0.2,var_limit=(0,0.5)),
        MotionBlur(p=0.2),
        RandomResizedCrop(height=cropsize[0], width=cropsize[1], always_apply=True)
        # RandomCrop(height=cropsize[0], width=cropsize[1], always_apply=True),
    ], p=p)




def create_transformer(transformations, images):
    target = {}
    return albu.Compose(transformations, p=0.5, additional_targets=target)(image=images[0],
                                                                           mask=images[1],
                                                                           )
def create_target_transformer(transformations, images):
    target = {'mask0': 'mask'}
    return albu.Compose(transformations, p=0.5, additional_targets=target)(image=images[0],
                                                                           mask=images[1],
                                                                           mask0=images[2]
                                                                           )


##########
def aug_img_lab(img, lab, cropsize, p=0.5):
    images = [img, lab]
    transformed = create_transformer([strong_aug(p=p, cropsize=cropsize)], images)
    return transformed['image'], transformed['mask']
##########
def aug_img_lab_count(img, lab, cropsize, p=0.5):
    images = [img, lab]
    transformed = create_transformer([count_aug(p=p, cropsize=cropsize)], images)
    return transformed['image'], transformed['mask']
##########
def aug_img_target_lab(img, lab, mask, cropsize, p=0.5):
    images = [img, lab, mask]
    transformed = create_target_transformer([strong_aug_exitcrop(p=p, cropsize=cropsize)], images)
    return transformed['image'], transformed['mask'],transformed['mask0']

##########
def aug_target_img_point(img, point, cropsize, p=0.5):
    images = [img, point]
    transformed = create_transformer(strong_aug(p=p, cropsize=cropsize), images)
    # transformed = create_transformer([strong_aug_exitcrop(p=p, cropsize=cropsize)], images)
    return transformed['image'], transformed['mask']



##########
def cropping(image, y, x, dim1, dim2):
    cropped_img = image[dim1:dim1+y, dim2:dim2+x]
    return cropped_img
##########
def cut_and_paste(image1, label1_connections, image2, label2_connections, cut_size=(256,256)):
    size = image2.shape
    image2_y_loc = randint(0, size[0] - cut_size[0])
    image2_x_loc = randint(0, size[1] - cut_size[1])

    image2_cut = cropping(image2, cut_size[0], cut_size[1], image2_y_loc, image2_x_loc)
    label2_cut = cropping(label2_connections, cut_size[0], cut_size[1], image2_y_loc, image2_x_loc)

    size = image1.shape
    image1_y_loc = randint(0, size[0] - cut_size[0])
    image1_x_loc = randint(0, size[1] - cut_size[1])

    image1[image1_y_loc:image1_y_loc + cut_size[0], image1_x_loc:image1_x_loc + cut_size[1]] = image2_cut
    label1_connections[image1_y_loc:image1_y_loc + cut_size[0], image1_x_loc:image1_x_loc + cut_size[1]] = label2_cut

    return image1, label1_connections

##########
def multifusion(image1, label1):
    h = image1.shape[0]
    w = image1.shape[1]
    # newimg = Image.new('L', (h, w))
    # newlab = Image.new('L', (h, w))

    transform = albu.Compose([
        albu.RandomResizedCrop(height=h, width=w, scale=(0.2, 0.8), always_apply=True, p=1)
    ])

    # result1_img = cv2.resize(image1, None, fx=0.5, fy=0.5)
    # result1_lab = cv2.resize(label1, None, fx=0.5, fy=0.5)
    result1_img = image1
    result1_lab = label1

    transformed2 = transform(image=image1, mask=label1)
    result2_img = transformed2["image"]
    result2_lab = transformed2["mask"]

    transformed3 = transform(image=image1, mask=label1)
    result3_img = transformed3["image"]
    result3_lab = transformed3["mask"]

    transformed4 = transform(image=image1, mask=label1)
    result4_img = transformed4["image"]
    result4_lab = transformed4["mask"]

    horizontal_concatimg1 = np.concatenate((result1_img, result2_img), axis=1)
    horizontal_concatimg2 = np.concatenate((result3_img, result4_img), axis=1)
    vertical_concatimg = np.concatenate((horizontal_concatimg1, horizontal_concatimg2))

    horizontal_concatlab1 = np.concatenate((result1_lab, result2_lab), axis=1)
    horizontal_concatlab2 = np.concatenate((result3_lab, result4_lab), axis=1)
    vertical_concatlab = np.concatenate((horizontal_concatlab1, horizontal_concatlab2))

    return vertical_concatimg, vertical_concatlab

##########
def random_mask(image1, label1, ratio):

    _, binary_image = cv2.threshold(label1.astype('uint8'), 0, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    mask = np.zeros_like(image1)

    # 随机选择要掩盖的连通域
    # num_to_mask = min(random.randint(1, 10), num_labels-1)  # 需要掩盖的连通域数量
    num_to_mask = int((num_labels-1)*ratio)
    mask_indices = np.random.choice(range(1, num_labels), num_to_mask, replace=False)

    for idx in mask_indices:
        mask[labels == idx] = 255

    img1 = cv2.bitwise_and(image1, cv2.bitwise_not(mask))
    lab1 = cv2.bitwise_and(label1, cv2.bitwise_not(mask))

    return img1, lab1
