import os
import os.path as osp
import numpy as np
import cv2
from torch.utils import data
from config import cfg
import skimage.io
import torchvision.transforms.functional as TF
from Utils import cornerDetect
import torch

root = cfg.DATASET.GF4_5
ignore_label = 255


def makeDataset(mode):
    all_tokens = []

    assert mode in ['train', 'val', 'train_tiny', 'val_tiny', 'test']
    image_path = osp.join(root, "img", mode)
    label_path = osp.join(root, "gt")

    img_tokens = os.listdir(image_path)
    img_tokens.sort()

    label_tokens = []

    for i in img_tokens:
        t = i.split("_")
        if len(t) > 2:
            sub_path = t[0] + "_" + t[1]
        else:
            sub_path = t[0]
        label_tokens.append(sub_path + "/" + i)

    assert len(img_tokens) == len(label_tokens), "The number of images is inconsistent with the number of labels"

    for img_token, label_token in zip(img_tokens, label_tokens):
        token = (osp.join(image_path, img_token), osp.join(label_path, label_token))
        all_tokens.append(token)

    print(f'GF4_5 has a total of {len(all_tokens)} images in {mode} phase')

    return all_tokens


class GF4_5Dataset(data.Dataset):

    def __init__(self, quality, mode, transform, edge_map=True, corner_map=True, thicky=1, dump=False):

        self.quality = quality
        self.mode = mode
        self.transform = transform
        self.edge_map = edge_map
        self.corner_map = corner_map
        self.thicky = thicky
        self.data_tokens = makeDataset(mode)
        self.dump = dump
        assert len(self.data_tokens), "the data is empty!Please check the root"

    def dump_images(self, image_name, image=None, label=None, boundary=None, body=None, corner=None):
        outdir = './dump_imgs_{}'.format(self.mode)
        os.makedirs(outdir, exist_ok=True)
        out_img_fn = os.path.join(outdir, image_name + '.png')
        out_label_fn = os.path.join(outdir, image_name + '_label.png')
        out_body_fn = os.path.join(outdir, image_name + '_body.png')
        out_boundary_fn = os.path.join(outdir, image_name + '_boundary.png')

        out_corner_fn = os.path.join(outdir, image_name + "_corner.png")
        if image is not None:
            image = TF.to_pil_image(image)
            image.save(out_img_fn)

        if label is not None:
            label = TF.to_pil_image(label)
            label.save(out_label_fn)
        if boundary is not None:
            boundary = TF.to_pil_image(boundary)
            boundary.save(out_boundary_fn)
        if body is not None:
            body = TF.to_pil_image(body)
            body.save(out_body_fn)

        if corner is not None:
            corner = TF.to_pil_image(corner)
            corner.save(out_corner_fn)

    def __getitem__(self, index):
        token = self.data_tokens[index]
        image_path, label_path = token
        image_name = osp.splitext(osp.basename(image_path))[0]

        image, label = skimage.io.imread(image_path), skimage.io.imread(label_path)
        image = image[:, :, :3]

        image = TF.to_tensor(image.astype(np.float32))
        image = TF.to_pil_image(image)
        label = TF.to_pil_image(label)
        if self.transform is not None:
            if self.mode in ["test", "val", 'val_tiny', 'test_tiny']:
                image, label = self.transform(image, label, size=[256, 256])
                label = torch.where(label != 0, torch.tensor(1.0), torch.tensor(0.0))
            else:
                image, label = self.transform(image, label, ["roate", "vflipAndhflip", "pad"], size=[256, 256])
                label = torch.where(label != 0, torch.tensor(1.0), torch.tensor(0.0))
        if self.dump:
            self.dump_images(image_name, image=image, label=label)
        result = {
            "image": image,
            "label": label,
            "image_name": image_name
        }
        if self.edge_map:
            boundary = self.get_boundary(label, thicky=self.thicky)
            body = self.get_body(label, boundary)
            if self.dump:
                self.dump_images(image_name, boundary=boundary, body=body)
            edge_map_dict = {
                "boundary": boundary,
                "body": body
            }
            result.update(edge_map_dict)

        if self.corner_map:
            corners = self.get_corner(label, label.shape[1:])
            corner_dict = {
                "corner": corners
            }
            result.update(corner_dict)
            if self.dump:
                self.dump_images(image_name, corner=corners)
        return result

    def __len__(self):
        return len(self.data_tokens)

    @staticmethod
    def get_boundary(mask, thicky=8):
        tmp = mask.data.numpy().astype('uint8')
        tmp = tmp[0]  # from [1,w,h] to [w,h]

        contour, _, = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        boundary = boundary.astype(np.float)
        boundary = TF.to_tensor(boundary)
        return boundary

    @staticmethod
    def get_body(mask, edge):
        edge_valid = edge == 1
        body = mask.clone()
        body[edge_valid] = ignore_label
        return body

    @staticmethod
    def get_corner(label, size):
        label = label.data.numpy()
        label = np.where(label == 1, 255, label)
        label = label[0]  # [1,h,w]->[h,w]
        label.astype(np.uint8)
        instance_gt = cornerDetect.InstanceGenerator(label)
        size = size
        corners = cornerDetect.getCorners(instance_gt, size)
        corner_label = cornerDetect.corners_to_img(size, corners)
        corner_label = TF.to_tensor(corner_label)
        return corner_label


if __name__ == '__main__':
    all = makeDataset("train")
    for i in range(len(all)):
        try:
            img = skimage.io.imread(all[i][0])
            gt = skimage.io.imread(all[i][1])
        except:
            print(all[i])

    print(0)
