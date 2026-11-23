import math
import numpy as np
import torch
import random
import cv2
import numpy


class Scale(object):
    """Resize image and label to a fixed size"""

    def __init__(self, wi, he):
        self.w = wi
        self.h = he

    def __call__(self, img, label):
        img = cv2.resize(img, (self.w, self.h))
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return [img, label]


class RandomCropResize(object):
    """Random crop followed by resize back to original size"""

    def __init__(self, crop_area):
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            top, bottom = random.randint(0, self.ch), random.randint(0, self.ch)
            left, right = random.randint(0, self.cw), random.randint(0, self.cw)

            img_crop = img[top:h - bottom, left:w - right]
            label_crop = label[top:h - bottom, left:w - right]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
            return img_crop, label_crop

        return [img, label]


class RandomFlip(object):
    """Random horizontal and vertical flip"""

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return [image, label]


class RandomExchange(object):
    """Randomly swap pre- and post-event images (for bi-temporal input)"""

    def __call__(self, image, label):
        if random.random() < 0.5:
            pre_img = image[:, :, 0:3]
            post_img = image[:, :, 3:6]
            image = numpy.concatenate((post_img, pre_img), axis=2)
        return [image, label]


class RandomRotate90(object):
    """Apply one shared quarter-turn rotation to both dates and the mask."""

    def __call__(self, image, label):
        k = random.randrange(4)
        return [np.rot90(image, k).copy(), np.rot90(label, k).copy()]


class RandomPhotometric(object):
    """Independent per-date photometric jitter.

    The TAR bridge reads |P - Q|, so a global exposure change between the two
    dates lands in the difference stream as if it were a change.  Applying
    brightness/contrast/gamma independently to each date forces that stream to
    separate a gain change from a real change; with probability
    ``shared_probability`` both dates get the same gain, which must NOT look
    like a change.  Pixels stay in 0..255 so Normalize can follow unchanged.
    """

    def __init__(self, probability=0.8, brightness=0.2, contrast=0.25,
                 gamma=0.2, shared_probability=0.25):
        self.probability = probability
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.shared_probability = shared_probability

    def _sample(self):
        brightness = random.uniform(-self.brightness, self.brightness)
        contrast = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        gamma = math.exp(random.uniform(-self.gamma, self.gamma))
        return brightness, contrast, gamma

    def __call__(self, image, label):
        if random.random() >= self.probability:
            return [image, label]
        image = image.astype(np.float32)
        shared = random.random() < self.shared_probability
        gains = self._sample(), self._sample()
        output = image.copy()
        for date in range(2):
            brightness, contrast, gamma = gains[0] if shared else gains[date]
            channels = slice(date * 3, (date + 1) * 3)
            block = np.clip(image[:, :, channels] / 255.0, 0.0, 1.0) ** gamma
            mean = block.mean()
            block = (block - mean) * contrast + mean + brightness
            output[:, :, channels] = np.clip(block, 0.0, 1.0) * 255.0
        return [output, label]


class Normalize(object):
    """Normalize both RGB dates and binarize the 0/255 change mask."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = image.astype(np.float32) / 255.0
        label = (label >= 128).astype(np.float32)

        for i in range(6):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]

        return [image, label]


class ToTensor(object):
    """Convert numpy arrays to PyTorch tensors"""

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (w, h))
            label = cv2.resize(
                label,
                (int(w / self.scale), int(h / self.scale)),
                interpolation=cv2.INTER_NEAREST
            )

        image = np.ascontiguousarray(image.transpose((2, 0, 1)))

        image_tensor = torch.from_numpy(image)
        label_tensor = torch.LongTensor(
            np.array(label, dtype=np.int_)
        ).unsqueeze(0)

        return [image_tensor, label_tensor]


class Compose(object):
    """Compose multiple transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
