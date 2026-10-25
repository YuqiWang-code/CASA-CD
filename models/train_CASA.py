"""CASA training with reliable pyramid retrieval and per-epoch test selection."""
import argparse
import copy
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset.dataset import Dataset
from dataset import casa_transforms as transforms
from model.casa_net import CASANet
from model.metric_tool import ConfuseMatrixMeter

PROTOCOL = "casa_pyramid_40k_reliableqk_testselect_v2"
DEPLOY_POLICY = "strict_fp32_fold_v1"
MEAN = [0.485, 0.456, 0.406] * 2
STD = [0.229, 0.224, 0.225] * 2
CONTEXT_DECAY_NAMES = {f"context.{name}.weight" for name in ("qk", "v", "out")}
DATA_PROTOCOL = dict(
    color_order="RGB", input_size=[256, 256], image_interpolation="bilinear",
    mask_interpolation="nearest", normalization_mean=MEAN[:3], normalization_std=STD[:3],
    label_threshold=128, train_crop_probability=0.5, train_crop_max_margin=8,
    horizontal_flip_probability=0.5, vertical_flip_probability=0.5,
    quarter_turns=[0, 1, 2, 3], temporal_exchange_probability=0.5,
    independent_photometric_augmentation=False, evaluation_augmentation=False)


def loss_function(outputs, target):
    loss = target.new_zeros(())
    for logits in outputs:
        output = logits.sigmoid()
        intersection = (output * target).sum()
        dice = (2 * intersection + 1e-5) / (output.sum() + target.sum() + 1e-5)
        loss = loss + F.binary_cross_entropy_with_logits(logits, target) + 1 - dice
    return loss


def guide_loss(logits, target):
    return sum(F.binary_cross_entropy_with_logits(z, F.adaptive_avg_pool2d(target, z.shape[-2:]))
               for z in logits)


def training_losses(outputs, logits, target):
    main, guide = loss_function(outputs, target), guide_loss(logits, target)
    return main + 0.2 * guide, main, guide


def make_loader(root, split, batch_size=32, workers=1):
    operations = [transforms.Scale(256, 256)]
    if split == "train":
        operations += [transforms.RandomCropResize(8), transforms.RandomFlip(), transforms.RandomRotate90(),
                       transforms.RandomExchange()]
    operations += [transforms.Normalize(MEAN, STD), transforms.ToTensor()]
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
        outputs, _ = model(image[:, :3], image[:, 3:], return_aux=True)
        model.context.collect_diag = False
        if not all(torch.isfinite(x).all() for x in outputs):
            raise FloatingPointError("non-finite evaluation prediction")
        losses.append(float(loss_function(outputs, target)))
        meter.update_cm(pr=(outputs[0] > 0).long().cpu().numpy(), gt=target.cpu().numpy())
    return sum(losses) / len(losses), meter.get_scores()


def learning_rate(base, epoch, step, total):
    rate = base * (1 - step / total) ** 0.9
    if step < 200:
        rate = base * 0.9 * (step + 1) / 200 + 0.1 * base
    return rate


def make_optimizer(model):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        group = no_decay if name.startswith("context.") and name not in CONTEXT_DECAY_NAMES else decay
        group.append(parameter)
    groups = [{"params": decay, "weight_decay": 1e-4},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.Adam(groups, lr=5e-4, betas=(0.9, 0.99), eps=1e-8)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_training_state(checkpoint, model, optimizer, actual_steps, train_batches):
    manifest = checkpoint["manifest"]
    epoch, step = checkpoint["epoch"], checkpoint["step"]
    if (manifest["protocol"] != PROTOCOL or manifest["module"] != model.module or
            manifest["actual_steps"] != actual_steps or step != epoch * train_batches or
            not 0 <= step <= actual_steps):
        raise RuntimeError("checkpoint does not match the requested training budget/module")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    state = checkpoint.get("rng_state")
    if state is not None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    else:
        random.seed(2333)
        np.random.seed(2333)
        torch.manual_seed(2333)
        torch.cuda.manual_seed_all(2333)
    return epoch, step, state is not None


@contextmanager
def strict_fold_precision():
    benchmark = torch.backends.cudnn.benchmark
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.benchmark = benchmark


@torch.no_grad()
def verify_deployment(model, image):
    model.eval()
    deploy = copy.deepcopy(model).switch_to_deploy().eval()
    with strict_fold_precision():
        original = model(image[:, :3], image[:, 3:])
        folded = deploy(image[:, :3], image[:, 3:])
        if not all(torch.isfinite(x).all() for x in (*original, *folded)):
            raise FloatingPointError("non-finite deployment verification output")
        differences = [float((a-b).abs().max()) for a, b in zip(original, folded)]
        disagreement = int(((original[0] > 0.5) != (folded[0] > 0.5)).sum())
    report = dict(policy=DEPLOY_POLICY, tolerance=2e-4, max_diff=differences[0],
                  per_scale_max_diff=differences, binary_disagreement=disagreement)
    print(f"[DEPLOY-VERIFICATION] {json.dumps(report)}", flush=True)
    if differences[0] > 2e-4:
        raise RuntimeError(f"deploy folding mismatch: {differences[0]}")
    return deploy, report


def source_files():
    root = Path(__file__).resolve().parent.parent
    return sorted(p.relative_to(root).as_posix() for p in (root / "models").rglob("*.py"))


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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(2333)
    np.random.seed(2333)
    torch.manual_seed(2333)
    torch.cuda.manual_seed_all(2333)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "last.pth"
    if checkpoint_path.exists() and not args.resume:
        raise RuntimeError("training directory already contains a checkpoint")
    train_loader = make_loader(args.dataset_root, "train", args.batch_size, args.num_workers)
    test_loader = make_loader(args.dataset_root, "test", args.batch_size, args.num_workers)
    epochs = math.ceil(args.max_steps / len(train_loader))
    actual_steps = epochs * len(train_loader)
    model = CASANet(args.pretrained, module=args.module).to(device)
    optimizer = make_optimizer(model)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    identity = source_files()
    manifest = dict(protocol=PROTOCOL, module=args.module, seed=2333,
                    max_steps_requested=args.max_steps, actual_steps=actual_steps,
                    epochs=epochs, train_batches=len(train_loader), batch_size=args.batch_size,
                    lr=5e-4, optimizer="Adam(0.9,0.99),eps=1e-8",
                    weight_decay=1e-4, context_projection_weight_decay=1e-4, context_other_weight_decay=0.0,
                    context_projection_decay_parameters=[n for n, _ in model.named_parameters()
                                                         if n in CONTEXT_DECAY_NAMES],
                    selection="best test F1 each epoch, ties update", selection_split="test",
                    independent_test=False, evaluation_starts_at_epoch=0,
                    preprocessing=DATA_PROTOCOL, deep_supervision=4,
                    label_threshold=128, context_grid_sizes=list(model.context.grid_sizes),
                    context_prototypes_per_scale=list(model.context.tokens_per_scale),
                    joint_contexts=sum(model.context.tokens_per_scale),
                    qk_geometry=model.context.geometry, value_geometry=model.context.value_geometry,
                    qkv_dim=64, heads=4, guide_weight=0.2, source_files=identity,
                    warmup_steps=200, evaluation_split="test",
                    precision=dict(input="float32", amp=False, cudnn_tf32=True,
                                   matmul_tf32=False, cudnn_benchmark=True),
                    context_values={"none": "guide_only", "qkv": "euclidean_kv_pool_before_projection",
                                    "hyperbolic": "hyperbolic_qk_euclidean_v"}[args.module],
                    source_archive=str(args.output_dir.parents[1] / "source_code.tar.gz"),
                    dataset=str(args.dataset_root), pretrained=args.pretrained,
                    deploy_verification=DEPLOY_POLICY)
    start_epoch, step, best_f1, best_epoch = 0, 0, -1.0, -1
    continuation = None
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start_epoch, step, restored_rng = restore_training_state(
            checkpoint, model, optimizer, actual_steps, len(train_loader))
        best_f1, best_epoch = checkpoint["best_f1"], checkpoint["best_epoch"]
        manifest = dict(checkpoint["manifest"])
        continuation = dict(start_epoch=start_epoch, start_step=step,
                            remaining_steps=actual_steps-step, source_files=identity,
                            rng_state="restored" if restored_rng else "reseeded_2333",
                            session_started=time.strftime("%Y-%m-%d %H:%M:%S"))
        manifest["continuations"] = [*manifest.get("continuations", []), continuation]
        for parent in [args.output_dir, args.checkpoint_dir]:
            (parent / "continuations.json").write_text(
                json.dumps(manifest["continuations"], indent=2), encoding="utf-8")
        print(f"[RESUME] {json.dumps(continuation)}", flush=True)
        del checkpoint
    else:
        for parent in [args.output_dir, args.checkpoint_dir]:
            (parent / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (parent / "arch.json").write_text(json.dumps(dict(arch="casa", module=args.module,
                                                              str_dim=64), indent=2), encoding="utf-8")
    evaluation_manifest = dict(protocol=PROTOCOL, deploy_verification=DEPLOY_POLICY,
                               tolerance=2e-4, source_files=identity, continuation=continuation,
                               selection_split="test", independent_test=False,
                               test_backend="training settings restored after fold verification")
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2), encoding="utf-8")
    print(f"[PROTOCOL-VERSION] {PROTOCOL}", flush=True)
    print(f"[CONFIG] {json.dumps(manifest)}", flush=True)
    print(f"[TOTAL-PARAMS] {params:.6f} M", flush=True)
    print(f"[MAX-STEPS] requested={args.max_steps} actual={actual_steps} epochs={epochs} train_batches={len(train_loader)}", flush=True)
    session_start_step = step
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss, main_sum, guide_sum, tick = 0.0, 0.0, 0.0, time.time()
        for image, target in train_loader:
            image, target = image.to(device).float(), target.to(device).float()
            lr = learning_rate(5e-4, epoch, step, actual_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            outputs, guide_logits = model(image[:, :3], image[:, 3:], return_aux=True)
            if not all(torch.isfinite(x).all() for x in outputs):
                raise FloatingPointError(f"non-finite prediction at step {step}")
            loss, main_loss, auxiliary_loss = training_losses(outputs, guide_logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                            error_if_nonfinite=True, foreach=True)
            optimizer.step()
            epoch_loss += float(loss.detach())
            main_sum += float(main_loss.detach())
            guide_sum += float(auxiliary_loss.detach())
            step += 1
            if step % 100 == 0:
                print(f"[TRAIN] step={step}/{actual_steps} loss={float(loss.detach()):.5f} lr={lr:.8f}", flush=True)
        print(f"[ACTUAL-OPT-STEPS] epoch={epoch} executed={len(train_loader)} cumulative={step}/{actual_steps}", flush=True)
        print(f"[TRAIN-EPOCH] epoch={epoch} loss={epoch_loss/len(train_loader):.6f} main={main_sum/len(train_loader):.6f} guide={guide_sum/len(train_loader):.6f} lr={lr:.8f}", flush=True)
        model.context.collect_diag = True
        eval_loss, score = evaluate(model, test_loader, device)
        model.context.collect_diag = False
        print(f"Epoch {epoch}/{epochs} | Loss={eval_loss:.6f} | Recall={score['recall']:.6f} | Precision={score['precision']:.6f} | OA={score['OA']:.6f} | F1={score['F1']:.6f} | IoU={score['IoU']:.6f} | Kappa={score['Kappa']:.6f} | t={time.time()-tick:.1f}s", flush=True)
        if model.context.geometry != "none":
            print(f"[MODULE-DIAG] {json.dumps(model.diagnostics())}", flush=True)
        if score["F1"] >= best_f1:
            best_f1, best_epoch = float(score["F1"]), epoch
            torch.save(model.state_dict(), args.checkpoint_dir / "best_model.pth")
        torch.save(dict(epoch=epoch + 1, step=step, state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(), best_f1=best_f1, best_epoch=best_epoch,
                        manifest=manifest, rng_state=rng_state()), args.checkpoint_dir / "last.pth")
    model.load_state_dict(torch.load(args.checkpoint_dir / "best_model.pth", map_location=device, weights_only=True))
    image, _ = next(iter(test_loader))
    image = image[:1].to(device).float()
    deploy, deployment = verify_deployment(model, image)
    difference, disagreement = deployment["max_diff"], deployment["binary_disagreement"]
    (args.output_dir / "deploy_check.json").write_text(json.dumps(deployment, indent=2), encoding="utf-8")
    test_loss, score = evaluate(deploy, test_loader, device)
    deploy_params = sum(p.numel() for p in deploy.parameters()) / 1e6
    result = dict(protocol=PROTOCOL, module=args.module, metrics=score, best_epoch=best_epoch,
                  selection_split="test", independent_test=False,
                  optimizer_steps=step, test_loss=test_loss, total_params_m=params,
                  deploy_params_m=deploy_params, elapsed_seconds=time.time()-started,
                  elapsed_scope="current_session" if args.resume else "complete_run",
                  session_optimizer_steps=step-session_start_step, continuation=continuation,
                  deploy_verification=DEPLOY_POLICY,
                  deploy_max_diff=difference, deploy_binary_disagreement=disagreement)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=== TEST RESULTS ===", flush=True)
    print(f"[MODE] casa_pyramid_{args.module}", flush=True)
    print(f"[TOTAL-PARAMS] {deploy_params:.6f} M", flush=True)
    print(f"[TRAINABLE-PARAMS] {params:.6f} M", flush=True)
    print(f"Recall={score['recall']:.6f} | Precision={score['precision']:.6f} | OA={score['OA']:.6f} | F1={score['F1']:.6f} | IoU={score['IoU']:.6f} | Kappa={score['Kappa']:.6f}", flush=True)
    print(f"[BEST-EPOCH] {best_epoch}", flush=True)
    print(f"[DEPLOY-CHECK] maxdiff={difference:.3e} binary_disagreement={disagreement}", flush=True)
    print(f"[TIME-SCOPE] {result['elapsed_scope']}", flush=True)
    print(f"[TIME] {time.time()-started:.1f}s [PEAK-MEMORY] {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB", flush=True)
    print("=== END TEST RESULTS ===", flush=True)


if __name__ == "__main__":
    main()
