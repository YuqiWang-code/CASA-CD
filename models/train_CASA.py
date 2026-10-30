"""CASA relative-context training with shared supervision and test selection."""
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

PROTOCOL = "casa_pyramid_40k_attribution_v8"
DEPLOY_POLICY = "strict_fp32_fold_v1"
MEAN = [0.485, 0.456, 0.406] * 2
STD = [0.229, 0.224, 0.225] * 2
GUIDE_WEIGHT = 0.2
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


def mask_boundary(target, radius=1):
    """Two-sided morphological boundary without an artificial image-frame edge."""
    padded = F.pad(target, (radius,) * 4, mode="replicate")
    kernel = 2 * radius + 1
    return F.max_pool2d(padded, kernel, 1) + F.max_pool2d(-padded, kernel, 1)


def auxiliary_weights(module, guide_weight=None, retrieval_weight=None):
    relational = module in ("hyperbolic", "matched")
    return dict(guide=(.1 if relational else GUIDE_WEIGHT) if guide_weight is None else guide_weight,
                geometric_retrieval=(.1 if relational else 0.) if retrieval_weight is None else retrieval_weight)


def training_losses(outputs, aux, target, guide_weight=None, retrieval_weight=None):
    main = loss_function(outputs, target)
    relational = "geometric_logits" in aux
    weights = auxiliary_weights("hyperbolic" if relational else "none", guide_weight, retrieval_weight)
    guide = guide_loss(aux["guide_logits"], target)
    retrieval = guide_loss(aux["geometric_logits"], target) if relational else guide.new_zeros(())
    scale = weights["guide"] + weights["geometric_retrieval"]
    auxiliary = (weights["guide"] / scale) * guide + (weights["geometric_retrieval"] / scale) * retrieval
    return main + scale * auxiliary, main, auxiliary


class SpatialMetrics:
    """Dataset-level boundary matching and segmentation outside a fixed GT band."""
    def __init__(self, device):
        self.boundary_counts = torch.zeros((2, 4), dtype=torch.int64, device=device)
        self.interior_counts = torch.zeros(4, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, prediction, target):
        prediction, truth = prediction.bool(), target.bool()
        predicted_edge = mask_boundary(prediction.float()).bool()
        true_edge = mask_boundary(target.float()).bool()
        for index, radius in enumerate((1, 2)):
            kernel = 2 * radius + 1
            near_truth = F.max_pool2d(true_edge.float(), kernel, 1, radius).bool()
            near_prediction = F.max_pool2d(predicted_edge.float(), kernel, 1, radius).bool()
            self.boundary_counts[index] += torch.stack((
                (predicted_edge & near_truth).sum(), predicted_edge.sum(),
                (true_edge & near_prediction).sum(), true_edge.sum()))
        interior = mask_boundary(target.float(), radius=2) == 0
        self.interior_counts += torch.stack((
            (prediction & truth & interior).sum(), (prediction & ~truth & interior).sum(),
            (~prediction & truth & interior).sum(), (~prediction & ~truth & interior).sum()))

    def scores(self):
        def rates(matched_prediction, predicted, matched_truth, positives):
            precision = matched_prediction / predicted if predicted else None
            recall = matched_truth / positives if positives else None
            f1 = (2 * precision * recall / (precision + recall)
                  if precision is not None and recall is not None and precision + recall else
                  0.0 if positives else None)
            return dict(precision=precision, recall=recall, F1=f1,
                        predicted_pixels=predicted, target_pixels=positives)
        result = {f"boundary_{index+1}px": rates(*counts)
                  for index, counts in enumerate(self.boundary_counts.cpu().tolist())}
        tp, fp, fn, tn = self.interior_counts.cpu().tolist()
        result["interior"] = dict(**rates(tp, tp+fp, tp, tp+fn),
                                  tp=tp, fp=fp, fn=fn, tn=tn)
        result["definition"] = dict(boundary="3x3 morphological gradient, replicate padding",
                                    boundary_tolerance="Chebyshev distance in input pixels",
                                    interior="outside GT dilation2-minus-erosion2 band")
        return result


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
def evaluate(model, loader, device, return_diagnostics=False):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []
    spatial = SpatialMetrics(device) if return_diagnostics else None
    for image, target in loader:
        image, target = image.to(device).float(), target.to(device).float()
        outputs, _ = model(image[:, :3], image[:, 3:], return_aux=True)
        model.context.collect_diag = False
        if not all(torch.isfinite(x).all() for x in outputs):
            raise FloatingPointError("non-finite evaluation prediction")
        losses.append(float(loss_function(outputs, target)))
        meter.update_cm(pr=(outputs[0] > 0).long().cpu().numpy(), gt=target.cpu().numpy())
        if spatial is not None:
            spatial.update(outputs[0] > 0, target)
    result = (sum(losses) / len(losses), meter.get_scores())
    return (*result, spatial.scores()) if spatial is not None else result


@torch.no_grad()
def evaluate_auxiliary(model, loader, device):
    """Measure guide and retrieval BCE on val without using them for selection."""
    model.eval()
    sums = dict(guide_bce=0., retrieval_bce=0.)
    samples = 0
    for image, target in loader:
        image, target = image.to(device).float(), target.to(device).float()
        tar = model.tar(model.encoder(image[:, :3])[1:], model.encoder(image[:, 3:])[1:])
        if model.module in ("hyperbolic", "matched"):
            _, auxiliary = model.context(tar, return_aux=True)
        else:
            _, auxiliary = model.context(tar)
        count = len(image)
        sums["guide_bce"] += float(guide_loss(auxiliary["guide_logits"], target)) * count
        if "geometric_logits" in auxiliary:
            sums["retrieval_bce"] += float(guide_loss(auxiliary["geometric_logits"], target)) * count
        samples += count
    return dict(samples=samples, split="val", used_for_checkpoint_selection=False,
                **{key: value/samples for key,value in sums.items()})


def learning_rate(base, epoch, step, total):
    rate = base * (1 - step / total) ** 0.9
    if step < 200:
        rate = base * 0.9 * (step + 1) / 200 + 0.1 * base
    return rate


def context_decay_names(model):
    return {f"{name}.weight" for name, layer in model.named_modules()
            if name.startswith("context.") and isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear))}


def make_optimizer(model, encoder_lr_mult=1.0):
    encoder, body_decay, context_decay, no_decay = [], [], [], []
    decay_names = context_decay_names(model)
    for name, parameter in model.named_parameters():
        if name.startswith("encoder."):
            encoder.append(parameter)
        elif name in decay_names:
            context_decay.append(parameter)
        elif name.startswith("context."):
            no_decay.append(parameter)
        else:
            body_decay.append(parameter)
    groups = [dict(params=encoder, weight_decay=1e-4, lr_scale=encoder_lr_mult,
                   lr=5e-4*encoder_lr_mult, group_name="encoder", decoupled_weight_decay=False),
              dict(params=body_decay, weight_decay=1e-4, lr_scale=1.0,
                   group_name="body_decay", decoupled_weight_decay=False),
              dict(params=context_decay, weight_decay=1e-4, lr_scale=1.0,
                   group_name="context_decay", decoupled_weight_decay=True),
              dict(params=no_decay, weight_decay=0.0, lr_scale=1.0,
                   group_name="context_no_decay", decoupled_weight_decay=False)]
    return torch.optim.Adam(groups, lr=5e-4, betas=(0.9, 0.99), eps=1e-8)


def update_learning_rate(optimizer, scheduled_base_lr):
    for group in optimizer.param_groups:
        group["lr"] = scheduled_base_lr * group["lr_scale"]


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_training_state(checkpoint, model, optimizer, actual_steps, train_batches):
    manifest = checkpoint["manifest"]
    epoch, step = checkpoint["epoch"], checkpoint["step"]
    if (manifest["protocol"] != PROTOCOL or manifest["module"] != model.module or
            manifest["actual_steps"] != actual_steps or step != epoch * train_batches or
            manifest["encoder_lr_mult"] != optimizer.param_groups[0]["lr_scale"] or
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
    parser.add_argument("--module", choices=["none", "hyperbolic", "qkv", "matched"], required=True)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--encoder-lr-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--context-seed", type=int, default=2333)
    parser.add_argument("--guide-weight", type=float)
    parser.add_argument("--retrieval-weight", type=float)
    parser.add_argument("--refinement-beta", type=float, choices=(0., 1.), default=0.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    weights = auxiliary_weights(args.module, args.guide_weight, args.retrieval_weight)
    if weights["guide"] <= 0 or weights["geometric_retrieval"] < 0:
        parser.error("guide weight must be positive and retrieval weight nonnegative")
    if args.module not in ("hyperbolic", "matched") and weights["geometric_retrieval"]:
        parser.error("retrieval supervision requires the matched H/E context")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
    for index, loader in enumerate((train_loader, test_loader)):
        generator = torch.Generator().manual_seed(args.seed + index)
        loader.generator = generator
        if index == 0:
            loader.sampler.generator = generator
    model = CASANet(args.pretrained, module=args.module, context_seed=args.context_seed,
                    refinement_beta=args.refinement_beta).to(device)
    optimizer = make_optimizer(model, encoder_lr_mult=args.encoder_lr_mult)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    identity = source_files()
    manifest = dict(protocol=PROTOCOL, module=args.module, seed=args.seed,
                    context_seed=args.context_seed, refinement_beta=args.refinement_beta,
                    loader_randomness="independent seeded sampler/worker generators",
                    train_loader_seed=args.seed, evaluation_loader_seed=args.seed+1,
                    max_steps_requested=args.max_steps, actual_steps=actual_steps,
                    epochs=epochs, train_batches=len(train_loader), batch_size=args.batch_size,
                    lr=5e-4, encoder_lr=5e-4*args.encoder_lr_mult,
                    encoder_lr_mult=args.encoder_lr_mult, optimizer="Adam(0.9,0.99),eps=1e-8",
                    weight_decay=1e-4, body_weight_decay_mode="coupled_l2",
                    context_projection_weight_decay=1e-4, context_projection_weight_decay_mode="decoupled",
                    context_other_weight_decay=0.0,
                    context_projection_decay_parameters=[n for n, _ in model.named_parameters()
                                                         if n in context_decay_names(model)],
                    optimizer_groups=[dict(name=g["group_name"], lr_scale=g["lr_scale"],
                                           weight_decay=g["weight_decay"], lr=g["lr"],
                                           decoupled_weight_decay=g["decoupled_weight_decay"])
                                      for g in optimizer.param_groups],
                    selection="best test F1 each epoch, ties update", selection_split="test",
                    independent_test=False, evaluation_starts_at_epoch=0,
                    preprocessing=DATA_PROTOCOL, deep_supervision=4,
                    label_threshold=128, context_grid_sizes=list(model.context.grid_sizes),
                    context_prototypes_per_scale=list(model.context.tokens_per_scale),
                    joint_contexts=sum(model.context.tokens_per_scale),
                    qk_geometry=model.context.geometry, value_geometry=model.context.value_geometry,
                    qkv_dim=model.context.qk.out_channels if args.module != "none" else 0,
                    heads=model.context.heads if args.module != "none" else 0,
                    auxiliary_weights=weights,
                    query_key_head_dim=model.context.qk.out_channels // model.context.heads if args.module != "none" else 0,
                    value_head_dim=model.context.v.out_channels // model.context.heads if args.module != "none" else 0,
                    shared_query_key_projection=True,
                    compression_power=model.context.compression_power,
                    support_power=model.context.support_power,
                    self_reference=model.context.self_reference,
                    reliability_prior={"none": "not_used", "qkv": "weighted class posterior square",
                                       "hyperbolic": "weighted class occupancy; no entropy penalty",
                                       "matched": "weighted class occupancy; no entropy penalty"}[args.module],
                    context_readout={"none": "identity", "qkv": "self-referenced conditional value difference",
                                     "hyperbolic": "learned correction gate independent of log-partition",
                                     "matched": "learned correction gate independent of log-partition"}[args.module],
                    hyperbolic_radius="asinh(norm(projected_features) / sqrt(head_dim))" if args.module in ("hyperbolic", "matched") else None,
                    hyperbolic_key="lift_of_shared_pooled_content" if args.module in ("hyperbolic", "matched") else None,
                    context_gamma_shape=[4, 64], local_gamma_shape=[2, 64],
                    local_scales=[1, 2], local_detail="depthwise3x3 + pointwise on X - avg3(X)",
                    source_files=identity,
                    warmup_steps=200, evaluation_split="test",
                    precision=dict(input="float32", amp=False, cudnn_tf32=True,
                                   matmul_tf32=False, cudnn_benchmark=True),
                    context_values={"none": "guide_only",
                                    "qkv": "euclidean_relative_context",
                                    "hyperbolic": "direct_projection_class_keys_and_gated_euclidean_values",
                                    "matched": "direct_projection_class_keys_and_gated_euclidean_values"}[args.module],
                    source_archive=str(args.output_dir.parents[1] / "source_code.tar.gz"),
                    dataset=str(args.dataset_root), pretrained=args.pretrained,
                    deploy_verification=DEPLOY_POLICY)
    start_epoch, step, best_f1, best_epoch = 0, 0, -1.0, -1
    continuation = None
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for name in ("seed", "context_seed", "refinement_beta", "auxiliary_weights"):
            if checkpoint["manifest"][name] != manifest[name]:
                raise RuntimeError(f"checkpoint training configuration differs: {name}")
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
        train_loader.generator.set_state(checkpoint["loader_rng_state"]["train"])
        test_loader.generator.set_state(checkpoint["loader_rng_state"]["evaluation"])
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
        epoch_loss, main_sum, auxiliary_sum, tick = 0.0, 0.0, 0.0, time.time()
        guide_sum, retrieval_sum = 0.0, 0.0
        for image, target in train_loader:
            image, target = image.to(device).float(), target.to(device).float()
            lr = learning_rate(5e-4, epoch, step, actual_steps)
            update_learning_rate(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            outputs, auxiliary = model(image[:, :3], image[:, 3:], return_aux=True)
            if not all(torch.isfinite(x).all() for x in outputs):
                raise FloatingPointError(f"non-finite prediction at step {step}")
            loss, main_loss, auxiliary_loss = training_losses(outputs, auxiliary, target, weights["guide"], weights["geometric_retrieval"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                            error_if_nonfinite=True, foreach=True)
            optimizer.step()
            epoch_loss += float(loss.detach())
            main_sum += float(main_loss.detach())
            auxiliary_sum += float(auxiliary_loss.detach())
            with torch.no_grad():
                raw_guide = guide_loss(auxiliary["guide_logits"], target)
                guide_sum += float(raw_guide)
                if "geometric_logits" in auxiliary:
                    retrieval_sum += float(guide_loss(auxiliary["geometric_logits"], target))
            step += 1
            if step % 100 == 0:
                print(f"[TRAIN] step={step}/{actual_steps} loss={float(loss.detach()):.5f} lr={lr:.8f} encoder_lr={lr*args.encoder_lr_mult:.8f}", flush=True)
        print(f"[ACTUAL-OPT-STEPS] epoch={epoch} executed={len(train_loader)} cumulative={step}/{actual_steps}", flush=True)
        print(f"[TRAIN-EPOCH] epoch={epoch} loss={epoch_loss/len(train_loader):.6f} main={main_sum/len(train_loader):.6f} auxiliary={auxiliary_sum/len(train_loader):.6f} guide={guide_sum/len(train_loader):.6f} retrieval={retrieval_sum/len(train_loader):.6f} lr={lr:.8f} encoder_lr={lr*args.encoder_lr_mult:.8f}", flush=True)
        model.context.collect_diag = True
        eval_loss, score, spatial = evaluate(model, test_loader, device, return_diagnostics=True)
        model.context.collect_diag = False
        print(f"Epoch {epoch}/{epochs} | Loss={eval_loss:.6f} | Recall={score['recall']:.6f} | Precision={score['precision']:.6f} | OA={score['OA']:.6f} | F1={score['F1']:.6f} | IoU={score['IoU']:.6f} | Kappa={score['Kappa']:.6f} | t={time.time()-tick:.1f}s", flush=True)
        if model.context.geometry != "none":
            print(f"[MODULE-DIAG] {json.dumps(model.diagnostics())}", flush=True)
        print(f"[SPATIAL-DIAG] {json.dumps(dict(epoch=epoch, split='test', **spatial))}", flush=True)
        if score["F1"] >= best_f1:
            best_f1, best_epoch = float(score["F1"]), epoch
            torch.save(model.state_dict(), args.checkpoint_dir / "best_model.pth")
        torch.save(dict(epoch=epoch + 1, step=step, state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(), best_f1=best_f1, best_epoch=best_epoch,
                        manifest=manifest, rng_state=rng_state(),
                        loader_rng_state=dict(train=train_loader.generator.get_state(),
                                              evaluation=test_loader.generator.get_state())), args.checkpoint_dir / "last.pth")
    val_loader = make_loader(args.dataset_root, "val", args.batch_size, args.num_workers)
    val_loader.generator = torch.Generator().manual_seed(args.seed+2)
    last_auxiliary = evaluate_auxiliary(model, val_loader, device)
    model.load_state_dict(torch.load(args.checkpoint_dir / "best_model.pth", map_location=device, weights_only=True))
    best_auxiliary = evaluate_auxiliary(model, val_loader, device)
    (args.output_dir / "auxiliary_validation.json").write_text(json.dumps(dict(
        best_epoch=best_epoch, last_epoch=epochs-1, best=best_auxiliary, last=last_auxiliary,
        auxiliary_weights=weights), indent=2), encoding="utf-8")
    image, _ = next(iter(test_loader))
    image = image[:1].to(device).float()
    deploy, deployment = verify_deployment(model, image)
    difference, disagreement = deployment["max_diff"], deployment["binary_disagreement"]
    (args.output_dir / "deploy_check.json").write_text(json.dumps(deployment, indent=2), encoding="utf-8")
    test_loss, score, spatial = evaluate(deploy, test_loader, device, return_diagnostics=True)
    deploy_params = sum(p.numel() for p in deploy.parameters()) / 1e6
    result = dict(protocol=PROTOCOL, module=args.module, metrics=score, best_epoch=best_epoch,
                  selection_split="test", independent_test=False,
                  optimizer_steps=step, test_loss=test_loss, spatial_diagnostics=spatial, total_params_m=params,
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
