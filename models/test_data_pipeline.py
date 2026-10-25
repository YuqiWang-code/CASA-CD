"""RGB/date ordering, synchronized augmentation and stable binary loss checks."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from dataset import casa_transforms as transforms
from train_CASA import MEAN, STD, learning_rate, loss_function, make_loader


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ("A", "B", "label", "list"):
            (root / name).mkdir()
        a = np.full((256, 256, 3), (17, 83, 211), dtype=np.uint8)
        b = np.full_like(a, (199, 67, 29))
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:180, 130:210] = 255
        cv2.imwrite(str(root / "A" / "pair.png"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(root / "B" / "pair.png"), cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(root / "label" / "pair.png"), mask)
        (root / "list" / "test.txt").write_text("pair.png\n")
        image, target = next(iter(make_loader(root, "test", batch_size=1, workers=0)))
        expected = (torch.tensor([17, 83, 211, 199, 67, 29]) / 255 - torch.tensor(MEAN)) / torch.tensor(STD)
        assert torch.allclose(image[0, :, 0, 0], expected, atol=1e-6)
        assert torch.equal(target[0, 0], torch.from_numpy((mask >= 128).astype(np.int64)))

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:19, 12:25] = 255
    pair = np.stack((mask, mask, mask, 255-mask, 255-mask, 255-mask), -1)
    with patch.object(transforms.random, "random", return_value=0.0), \
            patch.object(transforms.random, "randrange", return_value=1):
        image, target = transforms.Compose([
            transforms.RandomFlip(), transforms.RandomRotate90(), transforms.RandomExchange()
        ])(pair, mask)
    assert np.array_equal(image[:, :, 0], 255-target)
    assert np.array_equal(image[:, :, 3], target)
    with patch.object(transforms.random, "random", return_value=0.0), \
            patch.object(transforms.random, "randint", side_effect=[1, 2, 3, 4]):
        cropped, crop_target = transforms.RandomCropResize(8)(pair, mask)
    assert np.array_equal(crop_target, cv2.resize(mask[1:-2, 3:-4], (32, 32), interpolation=cv2.INTER_NEAREST))
    assert np.array_equal(cropped, cv2.resize(pair[1:-2, 3:-4], (32, 32)))

    logits = torch.tensor([[[[-100.0, 100.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0]]]])
    loss = loss_function((logits,), target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert (logits.grad.abs() > 0.4).all()
    moderate = torch.tensor([[[[-2.0, 1.0]]]])
    probability = moderate.sigmoid()
    expected = F.binary_cross_entropy(probability, target) + 1 - (
        2 * (probability * target).sum() + 1e-5) / (probability.sum() + target.sum() + 1e-5)
    assert torch.allclose(loss_function((moderate,), target), expected)
    assert learning_rate(5e-4, 1, 185, 40145) == 5e-4 * .9 * 186 / 200 + 5e-5
    print("RGB/date ordering, mask alignment, synchronized augmentation, stable logits loss and 200-step warmup PASS")


if __name__ == "__main__":
    main()
