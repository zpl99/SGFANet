
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from pycococreatortools import pycococreatortools

def InstanceGenerator(binary):
    # binary_gt = io.imread(binary_file)
    binary_gt = binary
    try:
        binary_gt = binary_gt[:,:,0]
    except:
        pass
    instance_gt, n = ndi.label(binary_gt)
    instance_gt = np.uint8(instance_gt)
    return instance_gt


def getCorners(annotation, size):
    ann_id = 1
    corner_list = []
    for n_instn in range(1, np.max(annotation) + 1):
        category_info = {'id': 1, 'is_crowd': 0}
        binary_mask = np.array(annotation == n_instn, dtype='uint8')
        annotation_info = pycococreatortools.create_annotation_info(ann_id, int(1), category_info, binary_mask,
                                                                    size, tolerance=2)
        ann_id = ann_id + 1
        if annotation_info != None:
            corners = annotation_info["segmentation"]
            # print(len(corners[0]))
            corner_list.append(corners[0])

    return corner_list


def corners_to_img(size, corner_list):
    corner_label = np.zeros(size)
    Neighbors = [(1, 1), (1, -1), (1, 0), (-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1)]
    for i in corner_list:
        for j in range(0, len(i), 2):
            corner_label[int(i[j + 1]), int(i[j])] = 1
            for neighbor in Neighbors:
                # 八邻域
                dr, dc = neighbor
                try: # 如果邻域溢出了，就跳过
                    corner_label[int(i[j + 1]) + dr, int(i[j]) + dc] = 1
                except:
                    pass

    # np.save("corner.npy",corner_label)
    # plt.matshow(corner_label)
    # plt.show()
    return corner_label



if __name__ == '__main__':
    instance_gt = InstanceGenerator(r"C:\Users\Dell\Desktop\Code\ReWrite\22978930_15_0_0.tif")
    img_path = r"C:\Users\Dell\Desktop\Code\ReWrite\22978930_15_0_0.tif"
    image = Image.open(img_path)
    size = image.size
    annotation = instance_gt
    corners = getCorners(annotation, image)
    corner_label = corners_to_img(size, corners)
