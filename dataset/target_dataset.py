import os.path as osp

from PIL import Image
from torch.utils import data

from dataset.data_aug import *
from utils.detmap import generate_gaussianmap, make_gaussian_map, getcenter_makegaussian
from utils.tools_self import *
from skimage import morphology,measure
import sys
sys.path.append('..')
import config
from skimage.exposure import match_histograms

sigma = config.get_value()


##########
class targetDataSet_train_step1_segadv(data.Dataset):
    def __init__(self, root_img, root_point, list_path, max_iters=None, iter_start=0, crop_size=[512, 512],
                 batch_size=1):

        self.root_img = root_img
        self.root_point = root_point
        self.list_path = list_path

        self.crop_size = crop_size
        self.max_iters = max_iters

        self.iter = iter_start

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) * batch_size / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            point_file = osp.join(self.root_point, name[:-4] + '.png')
            self.files.append({
                "img": img_file,
                "point": point_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles1 = self.files[index]
        rand_index = random.randint(0, len(self.files) - 1)
        datafiles2 = self.files[rand_index]

        image1 = Image.open(datafiles1["img"])
        image1 = np.asarray(image1, np.float32)
        image2 = Image.open(datafiles2["img"])
        image2 = np.asarray(image2, np.float32)

        point1 = Image.open(datafiles1["point"])
        point1 = np.asarray(point1, np.float32)
        point2 = Image.open(datafiles2["point"])
        point2 = np.asarray(point2, np.float32)

        cut_size = [256, 256]
        kernel = np.ones((cut_size[0], cut_size[1]), np.uint8)
        density_map = cv2.filter2D(point2, -1, kernel, borderType=0)

        density_map[:cut_size[0] // 2, :] = float("-inf")
        density_map[-cut_size[0] // 2:, :] = float("-inf")
        density_map[:, :cut_size[1] // 2] = float("-inf")
        density_map[:, -cut_size[1] // 2:] = float("-inf")

        row, col = np.where(density_map == np.max(density_map))

        center_index = random.randint(0, row.size - 1)
        center_row = row[center_index]
        center_col = col[center_index]

        image2_cut = center_cropping(image2, cut_size[0], cut_size[1], center_row, center_col)
        point2_cut = center_cropping(point2, cut_size[0], cut_size[1], center_row, center_col)

        density_map = cv2.filter2D(point1, -1, kernel, borderType=0)
        density_map[:cut_size[0] // 2, :] = float("inf")
        density_map[-cut_size[0] // 2:, :] = float("inf")
        density_map[:, :cut_size[1] // 2] = float("inf")
        density_map[:, -cut_size[1] // 2:] = float("inf")

        row, col = np.where(density_map == np.min(density_map))

        center_index = random.randint(0, row.size - 1)
        center_row = row[center_index]
        center_col = col[center_index]

        image1[center_row - cut_size[0] // 2:center_row + cut_size[0] // 2,
        center_col - cut_size[1] // 2:center_col + cut_size[1] // 2] = image2_cut
        point1[center_row - cut_size[0] // 2:center_row + cut_size[0] // 2,
        center_col - cut_size[1] // 2:center_col + cut_size[1] // 2] = point2_cut

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

        point1 = cv2.dilate(point1, dilate_kernel)

        image1 = min_max(image1, max=1, min=0)
        image_as_np, point_as_np = aug_target_img_point(image1, point1, self.crop_size)
        image_as_np = min_max(image_as_np, max=1, min=0)

        labels = measure.label(point_as_np.astype('uint8'), connectivity=1)

        properties = measure.regionprops(labels)
        sparsepoint = np.zeros_like(point_as_np).astype('float')
        for prop in properties:
            sparsepoint[int(prop.centroid[0]), int(prop.centroid[1])] = 1


        size = image_as_np.shape
        count_map_np = make_gaussian_map(sparsepoint, sigma=(sigma, sigma))

        count_map_np = np.expand_dims(count_map_np, axis=0)
        count_as_tensor = torch.from_numpy(count_map_np.astype("float32")).float()

        points_as_tensor = torch.from_numpy(sparsepoint.astype("float32")).float()

        image_as_np = np.expand_dims(image_as_np, axis=0)  # add additional dimension
        image_as_tensor = torch.from_numpy(image_as_np.astype("float32")).float()


        return image_as_tensor, points_as_tensor, count_as_tensor, np.array(size)

##########
class kashthuriDataSet_train_step2_entroy(data.Dataset):
    def __init__(self, root_img, root_label, root_point, root_gtpointlab,list_path, max_iters=None, iter_start = 0, crop_size=[512, 512], batch_size=1):
        self.root_img = root_img
        self.root_label = root_label
        self.root_gtpointlab = root_gtpointlab
        self.root_point = root_point
        self.list_path = list_path

        self.crop_size = crop_size
        self.max_iters = max_iters

        self.iter = iter_start

        # self.mean = 0.55193
        # self.std = 0.11998

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters)*batch_size / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name[:-4] + '.png')
            gtpointlab_file = osp.join(self.root_gtpointlab, name[:-4] + '.png')
            point_file = osp.join(self.root_point, name[:-4] +'.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "pointlab": gtpointlab_file,
                "point": point_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles1 = self.files[index]

        rand_index = random.randint(0, len(self.files) - 1)
        datafiles2 = self.files[rand_index]

        image1 = Image.open(datafiles1["img"])
        image1 = np.asarray(image1, np.float32)

        image2 = Image.open(datafiles2["img"])
        image2 = np.asarray(image2, np.float32)

        label1 = Image.open(datafiles1["label"])
        label1 = np.asarray(label1, np.float32)
        name1 = datafiles1["name"]

        label2 = Image.open(datafiles2["label"])
        label2 = np.asarray(label2, np.float32)

        pointlab1 = Image.open(datafiles1["pointlab"])
        pointlab1 = np.asarray(pointlab1, np.float32)

        pointlab2 = Image.open(datafiles2["pointlab"])
        pointlab2 = np.asarray(pointlab2, np.float32)

        point1 = Image.open(datafiles1["point"])
        point1 = np.asarray(point1, np.float32)

        point2 = Image.open(datafiles2["point"])
        point2 = np.asarray(point2, np.float32)

        cut_size = [256, 256]

        kernel = np.ones((cut_size[0], cut_size[1]), np.uint8)
        density_map = cv2.filter2D(point2, -1, kernel, borderType=0)

        density_map[:cut_size[0] // 2, :] = float("-inf")  # no systrmemic
        density_map[-cut_size[0] // 2:, :] = float("-inf")
        density_map[:, :cut_size[1] // 2] = float("-inf")
        density_map[:, -cut_size[1] // 2:] = float("-inf")
        # padding = cut_size[0] // 2
        # density_map = cv2.copyMakeBorder(density_map, padding, padding, padding, padding,
        #                                  borderType=cv2.BORDER_CONSTANT, value=float('inf'))

        row, col = np.where(density_map == np.max(density_map))  # max labels counts

        center_index = random.randint(0, row.size - 1)
        center_row = row[center_index]
        center_col = col[center_index]

        image2_cut = center_cropping(image2, cut_size[0], cut_size[1], center_row, center_col)
        label2_cut = center_cropping(label2, cut_size[0], cut_size[1], center_row, center_col)
        pointlab2_cut = center_cropping(pointlab2, cut_size[0], cut_size[1], center_row, center_col)

        kernel = np.ones((cut_size[0], cut_size[1]), np.uint8)
        density_map = cv2.filter2D(point1, -1, kernel, borderType=0)
        density_map[:cut_size[0] // 2, :] = float("inf")  # no systrmemic
        density_map[-cut_size[0] // 2:, :] = float("inf")
        density_map[:, :cut_size[1] // 2] = float("inf")
        density_map[:, -cut_size[1] // 2:] = float("inf")

        # density_map = cv2.copyMakeBorder(density_map, padding, padding, padding, padding, borderType=cv2.BORDER_CONSTANT, value=float('inf'))

        row, col = np.where(density_map == np.min(density_map))

        center_index = random.randint(0, row.size - 1)
        center_row = row[center_index]
        center_col = col[center_index]

        image1[center_row - cut_size[0] // 2:center_row + cut_size[0] // 2,
        center_col - cut_size[1] // 2:center_col + cut_size[1] // 2] = image2_cut
        label1[center_row - cut_size[0] // 2:center_row + cut_size[0] // 2,
        center_col - cut_size[1] // 2:center_col + cut_size[1] // 2] = label2_cut
        pointlab1[center_row - cut_size[0] // 2:center_row + cut_size[0] // 2,
        center_col - cut_size[1] // 2:center_col + cut_size[1] // 2] = pointlab2_cut

        image1 = min_max(image1, max=1, min=0)
        # image_as_np, label_as_np, flabel_as_np = aug_img_target_lab(image1, pointlab1, label1, self.crop_size)
        image_as_np, flabel_as_np, label_as_np = aug_img_target_lab(image1, label1, pointlab1, self.crop_size)
        image_as_np = min_max(image_as_np, max=1, min=0)

        size = image_as_np.shape

        # label_as_np_copy = label_as_np.copy()
        # label_as_np_copy[label_as_np_copy != 0] = 1

        # label_as_np_copy = morphology.remove_small_objects(label_as_np_copy.astype('bool'), min_size=128, connectivity=1)
        # label_as_np_copy = label_as_np_copy.astype('uint8') * 255

        points_map_np, count_map_np = generate_gaussianmap(label_as_np, sigma=(sigma, sigma))

        count_map_np = np.expand_dims(count_map_np, axis=0)
        count_as_tensor = torch.from_numpy(count_map_np.astype("float32")).float()

        points_as_tensor = torch.from_numpy(points_map_np.astype("float32")).float()

        # label_as_np_copy = label_as_np_copy / 255
        label_as_tensor = torch.from_numpy(label_as_np.astype("float32")).long()

        flabel_as_tensor = torch.from_numpy(flabel_as_np.astype("float32")).long()

        image_as_np = np.expand_dims(image_as_np, axis=0)  # add additional dimension
        image_as_tensor = torch.from_numpy(image_as_np.astype("float32")).float()

        return image_as_tensor, label_as_tensor, points_as_tensor, count_as_tensor, np.array(size), name1, flabel_as_tensor

##########
class targetDataSet_val(data.Dataset):
    def __init__(self, root_img, root_label, root_det, list_path, max_iters=None, crop_size=[512, 512], batch_size=1):

        self.root_img = root_img
        self.root_label = root_label
        self.root_det = root_det
        self.list_path = list_path
        self.crop_size = crop_size

        self.mean = 0.55193
        self.std = 0.11998

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) * batch_size / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name[:-4] + '.png')
            det_file = osp.join(self.root_det, name[:-4] + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "det": det_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        image = Image.open(datafiles["img"])
        label = Image.open(datafiles["label"])
        det = Image.open(datafiles["det"])

        image_as_np = np.asarray(image, np.float32)
        label_as_np = np.asarray(label, np.float32)
        det_as_np = np.asarray(det, np.float32)

        original_label = torch.from_numpy(np.asarray(label_as_np) / 255)

        img_shape = image_as_np.shape

        image_as_np = min_max(image_as_np, max=1, min=0)
        image_as_tensor = torch.Tensor(image_as_np)

        label_as_np = label_as_np / 255

        label_as_np = torch.from_numpy(label_as_np).long()
        det_as_np = torch.from_numpy(det_as_np).long()
        return image_as_tensor, label_as_np, original_label, det_as_np, name
##########
class targetDataSet_test(data.Dataset):
    def __init__(self, root_img, root_label, list_path, ignore_path=None, max_iters=None, crop_size=[512, 512]):

        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name[:-4] + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]
        name = datafiles["name"]

        image = Image.open(datafiles["img"])
        label = Image.open(datafiles["label"])

        image_as_np = np.asarray(image, np.float32)
        label_as_np = np.asarray(label, np.float32)

        original_label = np.asarray(label_as_np) / 255

        img_shape = image_as_np.shape

        image_as_np = min_max(image_as_np, max=1, min=0)

        image_as_tensor = torch.tensor(image_as_np)

        original_label = torch.from_numpy(original_label)

        label_as_np = label_as_np / 255

        label_as_np = torch.from_numpy(label_as_np).long()
        return image_as_tensor, label_as_np, original_label, np.array(img_shape), name
##########
class kplusDataSet_test(data.Dataset):
    def __init__(self, root_img, root_label, list_path, ignore_path=None, max_iters=None, crop_size=[512, 512]):

        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name[:-4] + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]
        name = datafiles["name"]
        image = Image.open(datafiles["img"])
        label = Image.open(datafiles["label"])
        image_as_np = np.asarray(image, np.float32)
        label_as_np = np.asarray(label, np.float32)
        points_map_np, detection_map_np = generate_gaussianmap(label_as_np, sigma=(61, 61))
        detectionmap_np = np.expand_dims(detection_map_np, axis=0)
        detectionmap_as_tensor = torch.from_numpy(detectionmap_np.astype("float32")).long()
        original_label = np.asarray(label_as_np) / 255
        img_shape = image_as_np.shape
        image_as_np = min_max(image_as_np, max=1, min=0)
        # top_size, bottom_size, left_size, right_size = (1632 - image_as_np.shape[0], 0, 1632 - image_as_np.shape[1], 0)
        # image_as_np = cv2.copyMakeBorder(image_as_np, top_size, bottom_size, left_size, right_size,
        #                                  borderType=cv2.BORDER_CONSTANT)
        image_as_tensor = torch.tensor(image_as_np)
        original_label = torch.from_numpy(original_label)

        label_as_np = label_as_np / 255

        label_as_np = torch.from_numpy(label_as_np).long()
        return image_as_tensor, original_label, original_label, detectionmap_as_tensor, np.array(img_shape), name


if __name__ == '__main__':
    pass
