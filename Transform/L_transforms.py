
import torchvision.transforms.functional as TF
import random
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def roate(image, mask):
    if random.random() > 0.5:
        angle = random.randint(-20, 20)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
    return image, mask


def vflipAndhflip(image, mask):
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    return image, mask


def pad(image, mask):
    if random.random() > 0.5:
        image = TF.pad(image, 10)
        mask = TF.pad(mask, 10)
    return image, mask


def data_transform_pipline(image, mask, pipline=None, size=None):
    if pipline is None:
        pipline = []

    if size is not None:
        image = TF.resize(image, size)
        mask = TF.resize(mask, size)

    for i in pipline:
        if random.random() > 0.8:
            continue
        if i == "roate":
            image, mask = roate(image, mask)
        elif i == "vflipAndhflip":
            image, mask = vflipAndhflip(image, mask)
        elif i == "color_jittering":
            colorjitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
            image = colorjitter(image)
        else:
            break
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    image = TF.normalize(image, mean=[0.33, 0.34, 0.30], std=[0.17, 0.165, 0.17])
    return image, mask

def data_transform_pipline_for_s2image(image, mask, pipline=None, size=None):
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0)

    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image)
    image = image.to(torch.float32)
    mask = mask.to(torch.float32)

    if size is not None:
        image = TF.resize(image, size)
        mask = TF.resize(mask, size)
    if pipline is not None:
        for i in pipline:
            if i == "roate":
                image, mask = roate(image, mask)
            elif i == "vflipAndhflip":
                image, mask = vflipAndhflip(image, mask)
            elif i == "color_jittering":
                colorjitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
                image_rgb = image[:3, :, :]
                image_nir = image[3, :, :].unsqueeze(0)
                image_rgb = colorjitter(image_rgb)
                image = torch.cat([image_rgb, image_nir], dim=0)
            else:
                break
    image = TF.normalize(image, mean=[0.14172366, 0.12568618, 0.12004076, 0.1804051],
                         std=[0.03957363, 0.04393258, 0.0611819, 0.0827849])



    mask[mask != mask] = 0
    image[image != image] = 0
    mask[mask == float('inf')] = 0
    mask[mask == float('-inf')] = 0
    image[image == float('inf')] = 0
    image[image == float('-inf')] = 0
    return image, mask

if __name__ == "__main__":
    image = cv2.imread(r"C:\Users\dell\Desktop\Code\MyCDCode\data\gaofen_tiny\images\train\1_1.png")
    label = cv2.imread(r"C:\Users\dell\Desktop\Code\MyCDCode\data\gaofen_tiny\gt\train\1_1_label.png")
    image, mask = roate(image, label)
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[1].imshow(mask)
    plt.show()
