import torch
import numpy as np

bce = torch.nn.BCEWithLogitsLoss(reduction='none')

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def _upscan(f):
    for i, fi in enumerate(f):
        if fi == np.inf: continue
        for j in range(1,i+1):
            x = fi+j*j
            if f[i-j] < x: break
            f[i-j] = x


def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    # same for the ground truth
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    # same for the intersection
    intersection = float(np.sum(intersection.flatten()))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value


def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    return dice_coefficient_numpy(pred, target)

def dice_coeff_2label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0

    return dice_coefficient_numpy(pred[:, 0, ...], target[:, 0, ...]), dice_coefficient_numpy(pred[:, 1, ...], target[:, 1, ...])


def DiceLoss(input, target):
    '''
    in tensor fomate
    :param input:
    :param target:
    :return:
    '''
    smooth = 1.
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def pixel_acc(pred, target):
    metric_disc = SegmentationMetric(2)
    metric_cup = SegmentationMetric(2)

    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0

    metric_disc.addBatch(pred[:, 1, :, :].flatten(), target[:, 1, :, :].flatten())
    metric_cup.addBatch(pred[:, 0, :, :].flatten(), target[:, 0, :, :].flatten())

    PA_disc = metric_disc.pixelAccuracy()
    PA_cup = metric_cup.pixelAccuracy()

    iou_disc = metric_disc.meanIntersectionOverUnion()
    iou_cup = metric_cup.meanIntersectionOverUnion()

    return PA_cup, PA_disc, iou_cup, iou_disc
