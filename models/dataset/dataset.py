import cv2
from os.path import join as osp
import numpy
import torch.utils.data


def read_list(list_path):
    """Read a list/*.txt file: one image filename per line.

    The server-side dataset layout is `<root>/{A,B,label}/` + `<root>/list/*.txt`.
    """
    with open(list_path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


class Dataset(torch.utils.data.Dataset):
    """Bi-temporal change detection dataset in A/B/label + list format.

    Args:
        file_root: dataset root containing A/ B/ label/ dirs
        list_path: path to list/*.txt with image filenames
        transform: Compose of Transforms
    """
    def __init__(self, file_root, list_path, transform=None):
        self.file_list = read_list(list_path)

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

        pre_image = cv2.cvtColor(cv2.imread(pre_image_name), cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_name, 0)
        post_image = cv2.cvtColor(cv2.imread(post_image_name), cv2.COLOR_BGR2RGB)

        img = numpy.concatenate((pre_image, post_image), axis=2)

        if self.transform:
            [img, label] = self.transform(img, label)

        return img, label
