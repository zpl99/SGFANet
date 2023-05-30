"""
Code is modified based on  https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py#L33
"""

import numpy as np


class Evaluator(object):

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  # (TP+TN)/(ALL)
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        MIoU = np.nanmean(MIoU)

        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, f"gt shape is {gt_image.shape}, while pre shape is {pre_image.shape}"

        # gt_image=gt_image.astype(np.int8)
        # pre_image = pre_image.astype(np.int8)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.TP = self.confusion_matrix[1, 1]
        self.FN = self.confusion_matrix[1, 0]
        self.FP = self.confusion_matrix[0, 1]
        self.TN = self.confusion_matrix[0, 0]

    def F1(self):
        return (2 * self.TP) / (2 * self.TP + self.FP + self.FN)

    def Precision(self):
        return self.TP / (self.TP + self.FP)

    def Recall(self):
        return self.TP / (self.TP + self.FN)

    def OA(self):
        return (self.TP + self.TN) / (self.TP + self.FP + self.TN + self.FN)

    def IoU(self):
        """只算前景(1)类的IoU"""
        return self.TP / (self.TP + self.FP + self.FN)

    def Kappa(self):
        pe_rows = np.sum(self.confusion_matrix, axis=0)
        pe_cols = np.sum(self.confusion_matrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        po = np.trace(self.confusion_matrix) / float(sum_total)
        return (po - pe) / (1 - pe)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':
    import skimage.io
    import os
    import torchvision.transforms.functional as TF

    # pre = skimage.io.imread(
    #     r"C:\Users\Dell\Desktop\result\Spacenet7_8mICTNet\global_monthly_2020_01_mosaic_L15-0571E-1075N_2287_3888_13_0_0.png")
    # gt = skimage.io.imread(r"F:\data\spacenet7\spacenet7_8meter\mask_8_meter_crop\global_monthly_2020_01_mosaic_L15-0571E-1075N_2287_3888_13_0_0.tif")
    # gt = gt[:, :, 0]
    # # gt = gt // 255
    # # gt = cv2.resize(gt,(256, 256))
    # gt = np.where(gt != 0, 1, 0)
    # pre = pre // 255
    # e = Evaluator(num_class=2)
    # e.add_batch(gt, pre)
    # print(e.IoU())
    from glob import glob

    e = Evaluator(num_class=2)
    gt = glob(r"G:\result_robust\val_gt\*.tif")
    pre = glob(r"G:\result_robust\v10\epochbest\iv10_EDSRUnet_eval1\*.png")
    for g, p in zip(gt, pre):
        gt_data = skimage.io.imread(g)
        # gt_data = gt_data[:, :, 0]
        gt_data = np.where(gt_data == 255, 1, 0)
        pre_data = skimage.io.imread(p)
        pre_data = np.where(pre_data == 255, 1, 0)
        e.add_batch(gt_data, pre_data)
    print(e.IoU())
    print(e.Recall())
    print(e.F1())
    print(e.OA())
    print(e.Precision())
    # print(e.TP / (e.TP + e.FP + e.FN + e.TN))
    # print(e.FP / (e.TP + e.FP + e.FN + e.TN))
    # print(e.FN / (e.TP + e.FP + e.FN + e.TN))
    # print(e.TN / (e.TP + e.FP + e.FN + e.TN))
    print(e.Kappa())
    # label_tokens = []
    # for i in img_tokens:
    #     t = i.split("_")
    #     if len(t) > 2:
    #         sub_path = t[0] + "_" + t[1]
    #     else:
    #         sub_path = t[0]
    #     label_tokens.append(sub_path + "/" + i)
    # for p, g in zip(img_tokens, label_tokens):
    #     pre = skimage.io.imread(rf'C:\Users\Dell\Desktop\result\SpacenetResult\Gaofen_Spacenet\{p}')
    #     gg = g.split(".")[0]
    #     gg = gg + ".tif"
    #     try:
    #         gt = skimage.io.imread(rf"F:\data\MiddleResolutionSegData\GF45_NEW\gt\{gg}")
    #     except:
    #         ggg = gg.split("/")[1]
    #         gt = skimage.io.imread(rf"F:\data\MiddleResolutionSegData\Spacenet7_4_meter_data\label\test\{ggg}")
    #         gt = gt[:,:,0]
    #     gt = cv2.resize(gt, (256, 256))
    #     gt = np.where(gt > 0, 1, 0)
    #     pre = np.where(pre==255,1,0)
    #     # print(gt.shape)
    #     # print(pre.shape)
    #     e.add_batch(pre, gt)
    # print(e.IoU())
    # print(e.Mean_Intersection_over_Union())
    # print(e.Pixel_Accuracy_Class())
    # print(e.OA())
