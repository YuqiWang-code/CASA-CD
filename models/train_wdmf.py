"""WDMF-Net training protocol for the CASA MobileNet/HFEA + TAR/DCR models."""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset.dataset import Dataset
from dataset import wdmf_transforms as transforms
from model.casa_wdmf_str_net import CASAWDMFSTRNet
from model.metric_tool import ConfuseMatrixMeter

PROTOCOL = "wdmf_str_40k_epochrounded_v1"
MEAN = [0.406, 0.456, 0.485] * 2
STD = [0.225, 0.224, 0.229] * 2


def loss_function(outputs, target):
    loss = target.new_zeros(())
    for output in outputs:
        intersection = (output * target).sum()
        dice = (2 * intersection + 1e-5) / (output.sum() + target.sum() + 1e-5)
        loss = loss + F.binary_cross_entropy(output, target) + 1 - dice
    return loss


def make_loader(root, split, batch_size=32, workers=1):
    operations = [transforms.Normalize(MEAN, STD), transforms.Scale(256, 256)]
    if split == "train":
        operations += [transforms.RandomCropResize(8), transforms.RandomFlip(),
                       transforms.RandomExchange()]
    operations.append(transforms.ToTensor())
    data = Dataset(file_root=str(root), list_path=str(root / "list" / f"{split}.txt"),
                   transform=transforms.Compose(operations))
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=split == "train",
                                       num_workers=workers, pin_memory=False,
                                       drop_last=split == "train")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []
    for image, target in loader:
        image, target = image.to(device).float(), target.to(device).float()
        outputs = model(image[:, :3], image[:, 3:])
        for op in model.stage_modules:
            op.collect_diag = False
        if not all(torch.isfinite(x).all() for x in outputs):
            raise FloatingPointError("non-finite evaluation prediction")
        losses.append(float(loss_function(outputs, target)))
        meter.update_cm(pr=(outputs[0] > 0.5).long().cpu().numpy(), gt=target.cpu().numpy())
    return sum(losses) / len(losses), meter.get_scores()


def learning_rate(base, epoch, step, total):
    rate = base * (1 - step / total) ** 0.9
    if epoch == 0 and step < 200:
        rate = base * 0.9 * (step + 1) / 200 + 0.1 * base
    return rate


def source_identity():
    root = Path(__file__).resolve().parent.parent
    paths = ["models/train_wdmf.py", "models/model/casa_wdmf_str_net.py",
             "models/model/wdmf_mobilenet_v2.py", "models/model/wdmf_encoder.py",
             "models/model/layers/change_context_qkv.py", "models/model/str_tar.py",
             "models/model/str_dcr.py", "models/model/str_reparam.py",
             "models/dataset/dataset.py", "models/dataset/wdmf_transforms.py",
             "models/model/metric_tool.py"]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--module", choices=["none", "hyperbolic", "qkv"], required=True)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(2333)
    np.random.seed(2333)
    torch.manual_seed(2333)
    torch.cuda.manual_seed_all(2333)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if (args.checkpoint_dir / "last.pth").exists():
        raise RuntimeError("training directory already contains a checkpoint")
    train_loader = make_loader(args.dataset_root, "train", args.batch_size, args.num_workers)
    validation_loader = make_loader(args.dataset_root, "val", args.batch_size, args.num_workers)
    test_loader = make_loader(args.dataset_root, "test", args.batch_size, args.num_workers)
    epochs = math.ceil(args.max_steps / len(train_loader))
    actual_steps = epochs * len(train_loader)
    model = CASAWDMFSTRNet(args.pretrained, module=args.module).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.99),
                                 eps=1e-8, weight_decay=1e-4)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    identity = source_identity()
    manifest = dict(protocol=PROTOCOL, module=args.module, seed=2333,
                    max_steps_requested=args.max_steps, actual_steps=actual_steps,
                    epochs=epochs, train_batches=len(train_loader), batch_size=args.batch_size,
                    lr=5e-4, optimizer="Adam(0.9,0.99),eps=1e-8,wd=1e-4",
                    selection="best validation F1, ties update", deep_supervision=4,
                    label_threshold=128, context_cell=4, qkv_dim=32, source_files=identity,
                    dataset=str(args.dataset_root), pretrained=args.pretrained)
    for parent in [args.output_dir, args.checkpoint_dir]:
        (parent / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (parent / "arch.json").write_text(json.dumps(dict(arch="casa_wdmf_str", module=args.module,
                                                          str_dim=64), indent=2), encoding="utf-8")
    print(f"[PROTOCOL-VERSION] {PROTOCOL}", flush=True)
    print(f"[CONFIG] {json.dumps(manifest)}", flush=True)
    print(f"[TOTAL-PARAMS] {params:.6f} M", flush=True)
    print(f"[MAX-STEPS] requested={args.max_steps} actual={actual_steps} epochs={epochs} train_batches={len(train_loader)}", flush=True)
    step, best_f1, best_epoch = 0, -1.0, -1
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        epoch_loss, tick = 0.0, time.time()
        for image, target in train_loader:
            image, target = image.to(device).float(), target.to(device).float()
            lr = learning_rate(5e-4, epoch, step, actual_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image[:, :3], image[:, 3:])
            if not all(torch.isfinite(x).all() for x in outputs):
                raise FloatingPointError(f"non-finite prediction at step {step}")
            loss = loss_function(outputs, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                            error_if_nonfinite=True, foreach=True)
            optimizer.step()
            epoch_loss += float(loss.detach())
            step += 1
            if step % 100 == 0:
                print(f"[TRAIN] step={step}/{actual_steps} loss={float(loss.detach()):.5f} lr={lr:.8f}", flush=True)
        print(f"[ACTUAL-OPT-STEPS] epoch={epoch} executed={len(train_loader)} cumulative={step}/{actual_steps}", flush=True)
        if epoch > 0:
            for op in model.stage_modules:
                op.collect_diag = True
            val_loss, score = evaluate(model, validation_loader, device)
            for op in model.stage_modules:
                op.collect_diag = False
            print(f"Epoch {epoch}/{epochs} | Loss={val_loss:.6f} | Recall={score['recall']:.6f} | Precision={score['precision']:.6f} | OA={score['OA']:.6f} | F1={score['F1']:.6f} | IoU={score['IoU']:.6f} | Kappa={score['Kappa']:.6f} | t={time.time()-tick:.1f}s", flush=True)
            if model.stage_modules:
                print(f"[MODULE-DIAG] {json.dumps(model.diagnostics())}", flush=True)
            if score["F1"] >= best_f1:
                best_f1, best_epoch = float(score["F1"]), epoch
                torch.save(model.state_dict(), args.checkpoint_dir / "best_model.pth")
        torch.save(dict(epoch=epoch + 1, step=step, state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(), best_f1=best_f1, best_epoch=best_epoch,
                        manifest=manifest), args.checkpoint_dir / "last.pth")
    model.load_state_dict(torch.load(args.checkpoint_dir / "best_model.pth", map_location=device, weights_only=True))
    model.eval()
    deploy = copy.deepcopy(model).switch_to_deploy().eval()
    image, _ = next(iter(validation_loader))
    image = image[:1].to(device).float()
    with torch.no_grad():
        original = model(image[:, :3], image[:, 3:])[0]
        folded = deploy(image[:, :3], image[:, 3:])[0]
        difference = float((original - folded).abs().max())
        disagreement = int(((original > 0.5) != (folded > 0.5)).sum())
    if difference > 2e-4:
        raise RuntimeError(f"deploy folding mismatch: {difference}")
    test_loss, score = evaluate(deploy, test_loader, device)
    deploy_params = sum(p.numel() for p in deploy.parameters()) / 1e6
    result = dict(protocol=PROTOCOL, module=args.module, metrics=score, best_epoch=best_epoch,
                  optimizer_steps=step, test_loss=test_loss, total_params_m=params,
                  deploy_params_m=deploy_params, elapsed_seconds=time.time()-started,
                  deploy_max_diff=difference, deploy_binary_disagreement=disagreement)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=== TEST RESULTS ===", flush=True)
    print(f"[MODE] casa_wdmf_str_{args.module}", flush=True)
    print(f"[TOTAL-PARAMS] {deploy_params:.6f} M", flush=True)
    print(f"[TRAINABLE-PARAMS] {params:.6f} M", flush=True)
    print(f"Recall={score['recall']:.6f} | Precision={score['precision']:.6f} | OA={score['OA']:.6f} | F1={score['F1']:.6f} | IoU={score['IoU']:.6f} | Kappa={score['Kappa']:.6f}", flush=True)
    print(f"[BEST-EPOCH] {best_epoch}", flush=True)
    print(f"[DEPLOY-CHECK] maxdiff={difference:.3e} binary_disagreement={disagreement}", flush=True)
    print(f"[TIME] {time.time()-started:.1f}s [PEAK-MEMORY] {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB", flush=True)
    print("=== END TEST RESULTS ===", flush=True)


if __name__ == "__main__":
    main()
