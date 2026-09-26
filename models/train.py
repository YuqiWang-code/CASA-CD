"""Train ChangeViT (Tiny/Small) baseline for binary remote-sensing change detection.

Adapted from the official ChangeViT main.py
(https://github.com/zhuduowang/ChangeViT). Training protocol is kept faithful to the
official repo:

- Loss: BCE + Dice (BCEDiceLoss)
- Optimizer: Adam(lr=2e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-4)
- Schedule: poly LR (power 0.9) over `max_steps`, with a 200-iteration linear warmup
- Budget: `max_steps` iterations (official 80000), batch 16, input 256x256
- Protocol: test set is used as the validation set; the best checkpoint is selected by
  test F1 (same as the official repo), and the final test is re-run on that checkpoint.

Adaptations for this repo (CASA-CD):
- A/B/label + list/*.txt dataset format (lab contract, label threshold gray>=128)
- Pretrained DeiT-Tiny / DINOv2 path passed via `--pretrained_weight_path`
- Logging: full config header + params/FLOPs at start; one line per epoch
  (loss + Recall/Precision/OA/F1/IoU/Kappa); final `=== TEST RESULTS ===` block
- Checkpoints: `last.pth` (resume, includes optimizer) + `best_F1=x.pth` (state dict)
"""
import os
import sys
import time
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Make the repo importable (train.py lives at models/ root).
_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.trainer import Trainer
from model.metric_tool import ConfuseMatrixMeter
from model.utils import BCEDiceLoss, init_seed, adjust_learning_rate

import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms


# -----------------------------------------------------------------------------
# Params / FLOPs measurement
# -----------------------------------------------------------------------------
def measure_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model, size=256):
    """Return flops (G) for a (1,3,size,size) bi-temporal pair (fvcore)."""
    from fvcore.nn import flop_count

    model.eval()
    pre = torch.randn(1, 3, size, size).cuda()
    post = torch.randn(1, 3, size, size).cuda()
    with torch.no_grad():
        counts, unsupported = flop_count(model, (pre, post))
    total = sum(counts.values())
    return total, len(unsupported)


def fmt_flops(total):
    # fvcore.flop_count already returns counts in giga-flops.
    return f"{total:.4f}"


def fmt_params(n):
    return f"{n / 1e6:.3f}"


# -----------------------------------------------------------------------------
# Train / val steps (faithful to the official ChangeViT main.py)
# -----------------------------------------------------------------------------
@torch.no_grad()
def val(args, val_loader, model):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    for iter, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        if args.onGPU:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = pre_img.float()
        post_img_var = post_img.float()
        target_var = target.float()

        output = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        epoch_loss.append(loss.data.item())
        salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_val, scores


def train_epoch(args, train_loader, model, optimizer, epoch, max_batches, cur_iter=0, lr_factor=1.):
    model.train()

    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    for iter, batched_inputs in enumerate(train_loader):
        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        # adjust the learning rate (poly + warmup, official schedule)
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches, lr_factor=lr_factor)

        if args.onGPU:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = pre_img.float()
        post_img_var = post_img.float()
        target_var = target.float()

        output = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())

        # computing F-measure on GPU, then accumulate the confusion matrix
        salEvalVal.update_cm(pr=pred.detach().cpu().numpy(), gt=target_var.detach().cpu().numpy())

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_train, scores, lr


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class ChangeViTTrainer(object):
    def __init__(self, args, log):
        self.args = args
        self.log = log

        init_seed(args.seed)
        torch.backends.cudnn.benchmark = True

        self.model = Trainer(
            args.model_type,
            pretrained_path=args.pretrained_weight_path,
            resnet_pretrained=args.resnet_pretrained,
        ).float()
        if args.onGPU:
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

        self.start_epoch = 0
        self.best_f1 = -1.0
        self.best_epoch = -1
        self.cur_iter = 0

        # auto-resume from ckpt_dir/last.pth (or an explicit --resume tar)
        resume_path = args.resume
        if resume_path is None:
            auto = os.path.join(args.ckpt_dir, "last.pth")
            if os.path.isfile(auto):
                resume_path = auto
        if resume_path is not None and os.path.isfile(resume_path):
            self._load_resume(resume_path)

        os.makedirs(args.ckpt_dir, exist_ok=True)

    def _load_resume(self, path):
        self.log(f"[RESUME] loading {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint.get("epoch", 0)
        self.best_f1 = checkpoint.get("best_f1", -1.0)
        self.best_epoch = checkpoint.get("best_epoch", -1)
        self.cur_iter = self.start_epoch * checkpoint.get("iters_per_epoch", 0)

    def _save_last(self, epoch, iters_per_epoch):
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "iters_per_epoch": iters_per_epoch,
        }, os.path.join(self.args.ckpt_dir, "last.pth"))

    def _make_loader(self, list_path, batch_size, shuffle):
        if shuffle:
            transform = myTransforms.Compose([
                myTransforms.Normalize(mean=self.args.mean, std=self.args.std),
                myTransforms.Scale(self.args.inWidth, self.args.inHeight),
                myTransforms.RandomCropResize(int(7. / 224. * self.args.inWidth)),
                myTransforms.RandomFlip(),
                myTransforms.RandomExchange(),
                myTransforms.ToTensor(),
            ])
        else:
            transform = myTransforms.Compose([
                myTransforms.Normalize(mean=self.args.mean, std=self.args.std),
                myTransforms.Scale(self.args.inWidth, self.args.inHeight),
                myTransforms.ToTensor(),
            ])
        dataset = myDataLoader.Dataset(file_root=self.args.dataset_root, list_path=list_path, transform=transform)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=self.args.num_workers, pin_memory=True, drop_last=False)

    def train(self):
        train_loader = self._make_loader(self.args.train_list, self.args.batch_size, True)
        test_loader = self._make_loader(self.args.test_list, self.args.test_batch_size, False)

        max_batches = len(train_loader)
        self.args.max_epochs = int(np.ceil(self.args.max_steps / max_batches))
        self.log(f"[MAX-STEPS] {self.args.max_steps}  [train_iters/epoch] {max_batches}  [max_epochs] {self.args.max_epochs}")
        self.log("EPOCH | Loss | Recall | Precision | OA | F1 | IoU | Kappa | t(s)")

        cur_iter = self.cur_iter
        for epoch in range(self.start_epoch, self.args.max_epochs):
            t0 = time.time()
            loss_tr, score_tr, lr = train_epoch(
                self.args, train_loader, self.model, self.optimizer, epoch, max_batches, cur_iter)
            cur_iter += max_batches

            torch.cuda.empty_cache()

            # official protocol: skip evaluation after the first epoch
            if epoch == 0:
                self._save_last(epoch, max_batches)
                continue

            loss_val, score_val = val(self.args, test_loader, self.model)
            torch.cuda.empty_cache()

            line = (f"Epoch {epoch}/{self.args.max_epochs} | Loss={loss_val:.4f} | "
                    f"Recall={score_val['recall']:.4f} | Precision={score_val['precision']:.4f} | "
                    f"OA={score_val['OA']:.4f} | F1={score_val['F1']:.4f} | "
                    f"IoU={score_val['IoU']:.4f} | Kappa={score_val['Kappa']:.4f} | "
                    f"t={time.time() - t0:.0f}s")
            self.log(line)

            # save last (resume) + best (test F1)
            self._save_last(epoch, max_batches)

            if score_val["F1"] > self.best_f1:
                prev_best = os.path.join(self.args.ckpt_dir, f"best_F1={self.best_f1:.4f}.pth")
                if self.best_f1 >= 0 and os.path.isfile(prev_best):
                    os.remove(prev_best)
                self.best_f1 = score_val["F1"]
                self.best_epoch = epoch
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.ckpt_dir, f"best_F1={self.best_f1:.4f}.pth"))

        self.log(f"[BEST] F1={self.best_f1:.4f} at epoch {self.best_epoch}")

    def test_best(self):
        best_path = os.path.join(self.args.ckpt_dir, f"best_F1={self.best_f1:.4f}.pth")
        if not os.path.isfile(best_path):
            self.log("[TEST] best checkpoint not found, skipping final test")
            return

        self.model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False))
        self.model = self.model.cuda()

        total_params = measure_params(self.model)
        trainable = measure_trainable_params(self.model)
        flops, n_unsup = measure_flops(self.model, size=self.args.inWidth)

        test_loader = self._make_loader(self.args.test_list, self.args.test_batch_size, False)
        _, score_test = val(self.args, test_loader, self.model)

        self.log("=== TEST RESULTS ===")
        self.log(f"[MODEL] ChangeViT-{self.args.model_type.upper()} baseline")
        self.log("[MODE] baseline")
        self.log(f"[TOTAL-PARAMS] {fmt_params(total_params)} M")
        self.log(f"[TRAINABLE-PARAMS] {fmt_params(trainable)} M")
        self.log(f"[FLOPS] {fmt_flops(flops)} G   (input 2x3x{self.args.inWidth}x{self.args.inHeight}, unsupported_ops={n_unsup})")
        self.log(f"Recall={score_test['recall']:.4f} | Precision={score_test['precision']:.4f} | OA={score_test['OA']:.4f} | "
                 f"F1={score_test['F1']:.4f} | IoU={score_test['IoU']:.4f} | Kappa={score_test['Kappa']:.4f}")
        self.log(f"[BEST-F1] {self.best_f1:.4f} (epoch {self.best_epoch})")
        self.log("=== END TEST RESULTS ===")


# -----------------------------------------------------------------------------
# Config header
# -----------------------------------------------------------------------------
def write_header(args, log):
    log("=" * 72)
    log(f"ChangeViT-{args.model_type.upper()} baseline  |  train_scripts/baseline/Run1")
    log("=" * 72)
    log("[CONFIG]")
    for k, v in vars(args).items():
        log(f"  {k}: {v}")
    log("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="ChangeViT baseline training (Run1)")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--test_list', type=str, required=True)
    parser.add_argument('--pretrained_weight_path', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=80000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--model_type', type=str, default='tiny', help='select vit model type | tiny | small')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Test batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs (step mode)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--seed', default=16, help='initialization seed number')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training (auto: ckpt_dir/last.pth)')
    parser.add_argument('--resnet_pretrained', type=int, default=1, help='load ImageNet ResNet18 weights for detail branch')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU id number')

    # official ChangeViT normalization (BGR order, ImageNet stats x2)
    parser.add_argument('--mean', type=float, nargs=6,
                        default=[0.406, 0.456, 0.485, 0.406, 0.456, 0.485])
    parser.add_argument('--std', type=float, nargs=6,
                        default=[0.225, 0.224, 0.229, 0.225, 0.224, 0.229])
    args = parser.parse_args()

    args.resnet_pretrained = bool(args.resnet_pretrained)
    if args.onGPU:
        torch.cuda.set_device(args.gpu_id)

    def log(msg):
        print(msg, flush=True)

    write_header(args, log)

    trainer = ChangeViTTrainer(args, log)

    log("[TOTAL-PARAMS] " + fmt_params(measure_params(trainer.model)) + " M")
    log("[TRAINABLE-PARAMS] " + fmt_params(measure_trainable_params(trainer.model)) + " M")
    flops, n_unsup = measure_flops(trainer.model, size=args.inWidth)
    log(f"[FLOPS] {fmt_flops(flops)} G   (input 2x3x{args.inWidth}x{args.inHeight}, unsupported_ops={n_unsup})")
    log("=" * 72)

    trainer.train()
    trainer.test_best()


if __name__ == "__main__":
    main()
