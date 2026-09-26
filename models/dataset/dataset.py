import cv2
import os
from os.path import join as osp
import numpy
import torch.utils.data


def read_list(list_path):
    """Read a list/*.txt file: one sample name per line (no extension).

    The server-side dataset layout is `<root>/{A,B,label}/` + `<root>/list/*.txt`.
    """
    with open(list_path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


class Dataset(torch.utils.data.Dataset):
    """Bi-temporal change detection dataset in A/B/label + list format.

    Args:
        file_root: dataset root containing A/ B/ label/ dirs
        list_path: path to list/*.txt with sample names (no extension)
        transform: Compose of Transforms
    """
    def __init__(self, file_root='data/', list_path=None, mode='train', transform=None):
        if list_path is not None and os.path.isfile(list_path):
            self.file_list = read_list(list_path)
        else:
            # legacy mode: file_root/<mode>/{A,B,label} with natural order
            self.file_list = os.listdir(osp(file_root, mode, 'A'))
            file_root = osp(file_root, mode)

        self.pre_images = [osp(file_root, 'A', x) for x in self.file_list]
        self.post_images = [osp(file_root, 'B', x) for x in self.file_list]
        self.gts = [osp(file_root, 'label', x) for x in self.file_list]

        self.transform = transform

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]

        pre_image = cv2.imread(pre_image_name)
        label = cv2.imread(label_name, 0)
        post_image = cv2.imread(post_image_name)

        img = numpy.concatenate((pre_image, post_image), axis=2)

        if self.transform:
            [img, label] = self.transform(img, label)

        return img, label

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
