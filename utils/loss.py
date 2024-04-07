import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.tools_self import *
import config

##########
class CrossEntropy2d(nn.Module):

    def __init__(self, reduction="mean", ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, predict, target, mask=None, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)

        if mask == None:
            loss = F.cross_entropy(predict, target, weight=weight, reduction=self.reduction)
            return loss
        else:
            loss = F.cross_entropy(predict, target, weight=weight, reduction="none")
            mask = mask.view(-1)
            nums = sum(mask)
            loss = torch.sum(mask * loss) / nums
            return loss

##########
def consistency_loss(pred, gt):
    loss = torch.nn.MSELoss()
    loss = loss(pred.float(), gt.float())
    return loss
##########
def loss_calc(pred, label, gpu, usecuda):
    if usecuda:
        label = label.long().cuda(gpu)
        criterion = CrossEntropy2d().cuda(gpu)
        result = criterion(pred, label)
    else:
        label = label.long()
        criterion = CrossEntropy2d()
        result = criterion(pred, label)

    return result
##########
def cross_entropy_2d_mask(predict, label, gpu):
    n, c, h, w = predict.size()
    label = label.long().cuda(gpu)
    target_mask = (label >= 0) * (label < 200)
    label = label[target_mask]
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(input=predict, target=label).cuda(gpu)
    return loss


##########
def weight_mse_source_seg(count_maps, points, counts, gpu):
    criterion = torch.nn.MSELoss(reduction='none').cuda(gpu)

    weight_maps = torch.zeros(count_maps.shape).cuda(gpu)
    wpoints = points.data.cpu().numpy()
    for i in range(weight_maps.shape[0]):
        weight_map = cv2.GaussianBlur(wpoints[i], (11, 11), 0, borderType=0)

        if np.amax(weight_map) != 0:
            weight_map /= np.amax(weight_map)
        weight_maps[i][0] = torch.from_numpy(weight_map).cuda(gpu)
    det_weight = torch.ones(count_maps.shape).cuda(gpu) + 3 * weight_maps

    num_mask = det_weight > 0
    nums = torch.sum(num_mask)

    loss_det = torch.sum(det_weight * criterion(count_maps, counts.float().cuda(gpu))) / nums

    return loss_det

##########
def weight_mse_source(count_maps, counts, gpu):
    criterion = torch.nn.MSELoss().cuda(gpu)
    loss_det = criterion(count_maps, counts.float().cuda(gpu))

    return loss_det



##########
def weight_mse_target(count_maps, count_semi, points, counts, gpu):

    criterion = torch.nn.MSELoss(reduction='none').cuda(gpu)

    weight_maps = torch.zeros(count_maps.shape).cuda(gpu)
    wpoints = points.data.cpu().numpy()
    for i in range(weight_maps.shape[0]):

        # weight_map = cv2.GaussianBlur(wpoints[i], (11, 11), 0, borderType=0)

        background_mask = (count_semi[i] == True).astype("float32")
        mask = torch.from_numpy(background_mask).cuda(gpu)
        mask[counts[i][0] > 0] = 1 # 1

        # if np.amax(weight_map) != 0:
        #     weight_map /= np.amax(weight_map)
        # weight_maps[i][0] = mask + 3 * torch.from_numpy(weight_map).cuda(gpu)
        weight_maps[i][0] = mask

    num_mask = weight_maps > 0
    nums = torch.sum(num_mask)
    loss_count = torch.sum(weight_maps * criterion(count_maps, counts.float().cuda(gpu))) / (nums+1)

    return loss_count

##########
class SCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a #两个超参数
        self.b = b


    def forward(self, pred, labels, gpu, usecuda):
        # CE 部分，正常的交叉熵损失
        ce = cross_entropy_2d_mask(pred, labels, gpu)

        # RCE
        # label = labels.long().cuda(gpu)
        # target_mask = (label >= 0) * (label < 200)
        # label[label>100] = 1
        # pred = F.softmax(pred, dim=1)
        # pred = torch.clamp(pred, min=1e-4, max=1.0)
        # pred = pred.transpose(1, 2).transpose(2, 3).contiguous()
        # label_one_hot = F.one_hot(label, self.num_classes).float().to(pred.device)
        # label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0) #最小设为 1e-4，即 A 取 -4
        # rce = (-1 * target_mask *torch.sum( pred * torch.log(label_one_hot), dim=3))
        #
        # loss = self.a * ce + self.b * rce.mean()
        loss = ce
        return loss

cofficient = standar_gaussian(config.get_value())
##########
def ranking_loss(target_pred, source_pred, gpu):

    loss5 = torch.nn.MarginRankingLoss(margin=0.0)
    loss6 = torch.nn.MarginRankingLoss(margin=0.0)

    predict = torch.sum(target_pred, dim=(2, 3)) / (torch.from_numpy(np.array(cofficient)).cuda(gpu))
    predict_source = torch.sum(source_pred, dim=(2, 3)) / (torch.from_numpy(np.array(cofficient)).cuda(gpu))

    tha = 3
    # tha = int(np.round(predict_source.squeeze().cpu().numpy() * 0.10))

    loss_5 = loss5(predict[0].float(), ((predict_source[0] - tha) * torch.ones((1,)).float().cuda(gpu)),
                   torch.ones((1,)).float().cuda(gpu))
    loss_6 = loss6(((predict_source[0] + tha) * torch.ones((1,)).float().cuda(gpu)), predict[0].float(),
                   torch.ones((1,)).float().cuda(gpu))
    return loss_5 + loss_6



if __name__ == "__main__":
    import torch

    count_map = torch.randn((2, 1, 512, 512))
    gt = torch.zeros((2, 1, 512, 512))
    gt[0, 0, 23, 67] = 255
    gt[0, 0, 255, 456] = 255

    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    loss4 = torch.nn.MSELoss()
    loss5 = torch.nn.MarginRankingLoss(margin=0.0)

    aggregation1 = F.adaptive_avg_pool2d(count_map, 1).squeeze(2).squeeze(2)
    gt_num = torch.sum(torch.sum(gt > 0, 2), 2)
    loss_4 = loss4(aggregation1.float(), gt_num.float())

    loss_5 = 0.1 * loss5(aggregation1, gt_num.float() * torch.ones((1,)).float(), torch.ones((1,)).float())

    index2 = gt != 0
    index3 = gt == 0
    loss_2 = loss2(count_map[index2], gt[index2])
    loss_3 = loss3(count_map[index3], gt[index3])
