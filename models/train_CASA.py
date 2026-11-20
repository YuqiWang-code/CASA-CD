"""CASA first-order context and foreground-interior loss controls."""
import argparse
import copy
import hashlib
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
from model.edge_supervision import edge_loss, sobel_edge_map
from model.metric_tool import ConfuseMatrixMeter
from operating_point import THRESHOLDS, ThresholdCounts

PROTOCOL = "casa_pyramid_40k_testselect_heads_detail_v12r3"
GUIDE_OBJECTIVE = "raw_area_bce"
DEPLOY_POLICY = "strict_fp32_fold_v1"
MEAN = [0.485, 0.456, 0.406] * 2
STD = [0.229, 0.224, 0.225] * 2
GUIDE_WEIGHT = 0.1
DATA_PROTOCOL = dict(
    color_order="RGB", input_size=[256, 256], image_interpolation="bilinear",
    mask_interpolation="nearest", normalization_mean=MEAN[:3], normalization_std=STD[:3],
    label_threshold=128, train_crop_probability=0.5, train_crop_max_margin=8,
    horizontal_flip_probability=0.5, vertical_flip_probability=0.5,
    quarter_turns=[0, 1, 2, 3], temporal_exchange_probability=0.5,
    independent_photometric_augmentation=False, evaluation_augmentation=False)


def loss_function(outputs, target, interior_positive_weight=1.0):
    loss = target.new_zeros(())
    for index, logits in enumerate(outputs):
        output = logits.sigmoid()
        intersection = (output * target).sum()
        dice = (2 * intersection + 1e-5) / (output.sum() + target.sum() + 1e-5)
        bce = foreground_interior_bce(logits, target, interior_positive_weight if index == 0 else 1.)
        loss = loss + bce + 1 - dice
    return loss


def guide_loss(logits, target):
    return sum(F.binary_cross_entropy_with_logits(z, F.adaptive_avg_pool2d(target, z.shape[-2:]))
               for z in logits)


def guide_terms(auxiliary, target):
    """Training guide objective and its separately named BCE components."""
    raw = guide_loss(auxiliary["guide_logits"], target)
    terms = dict(guide=raw, guide_raw=raw)
    return terms


def guide_effective_coefficients(module, guide_weight):
    return dict(raw=guide_weight, posterior=0.)


def mask_boundary(target, radius=1):
    """Two-sided morphological boundary without an artificial image-frame edge."""
    padded = F.pad(target, (radius,) * 4, mode="replicate")
    kernel = 2 * radius + 1
    return F.max_pool2d(padded, kernel, 1) + F.max_pool2d(-padded, kernel, 1)


def foreground_interior_bce(logits, target, positive_weight=1.0):
    """Normalize the first-output BCE by total pixel weight."""
    if positive_weight == 1.0:
        return F.binary_cross_entropy_with_logits(logits, target)
    interior_positive = target * (mask_boundary(target, 2) == 0)
    weight = 1 + (positive_weight - 1) * interior_positive
    return (weight * F.binary_cross_entropy_with_logits(logits, target, reduction="none")).sum() / weight.sum()


def auxiliary_weights(module, guide_weight=None, retrieval_weight=None):
    return dict(guide=GUIDE_WEIGHT if guide_weight is None else guide_weight,
                geometric_retrieval=0. if retrieval_weight is None else retrieval_weight)


def edge_terms(auxiliary, target, dilate=1, scales="all"):
    """Sobel edge loss over the edge maps the bridge or the retrieval exposes.

    ``scales="fine"`` keeps only the two finest levels (64^2 and 32^2), so the
    coarse maps still carry segmentation gradients but no edge target.
    """
    maps = auxiliary.get("edge_logits") or ()
    if not maps:
        return None
    if scales == "fine":
        maps = maps[:2] if len(maps) > 4 else maps[:2]
    edge = sobel_edge_map(target, dilate)
    losses = []
    for logits in maps:
        size = logits.shape[-2:]
        losses.append(edge_loss(logits, F.interpolate(edge, size=size, mode="area")))
    return torch.stack(losses).mean()


def training_losses(outputs, aux, target, guide_weight=None, retrieval_weight=None,
                    interior_positive_weight=1.0, edge_weight=0.0, edge_dilate=1,
                    edge_scales="all"):
    main = loss_function(outputs, target, interior_positive_weight)
    if "detail_logits" in aux:
        detail = aux["detail_logits"]
        detail_target = (F.adaptive_avg_pool2d(target, detail.shape[-2:]) if aux["detail_target"] == "area"
                         else F.interpolate(target, size=detail.shape[-2:], mode="nearest-exact"))
        main = main + loss_function((detail,), detail_target)
    relational = "geometric_logits" in aux
    weights = auxiliary_weights("hyperbolic" if relational else "none", guide_weight, retrieval_weight)
    guide = guide_terms(aux, target)["guide"]
    retrieval = (guide_loss(aux["geometric_logits"], target)
                 if relational and weights["geometric_retrieval"] > 0 else guide.new_zeros(()))
    scale = weights["guide"] + weights["geometric_retrieval"]
    auxiliary = (weights["guide"] / scale) * guide + (weights["geometric_retrieval"] / scale) * retrieval
    edges = edge_terms(aux, target, edge_dilate, edge_scales) if edge_weight > 0 else None
    total = main + scale * auxiliary
    if edges is not None:
        total = total + edge_weight * edges
    return total, main, auxiliary + (0.0 if edges is None else edge_weight * float(edges.detach()))


class SpatialMetrics:
    """Dataset-level boundary matching and segmentation outside a fixed GT band."""
    radii = (0, 1, 2, 3)

    def __init__(self, device):
        self.boundary_counts = torch.zeros((len(self.radii), 4), dtype=torch.int64, device=device)
        self.interior_counts = torch.zeros(4, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, prediction, target):
        prediction, truth = prediction.bool(), target.bool()
        predicted_edge = mask_boundary(prediction.float()).bool()
        true_edge = mask_boundary(target.float()).bool()
        for index, radius in enumerate(self.radii):
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
        result = {f"boundary_{radius}px": rates(*counts)
                  for radius, counts in zip(self.radii, self.boundary_counts.cpu().tolist())}
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
def evaluate(model, loader, device, return_diagnostics=False, operating_point=None,
             per_image=None):
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
        prediction = outputs[0] > 0
        meter.update_cm(pr=prediction.long().cpu().numpy(), gt=target.cpu().numpy())
        if per_image is not None:
            predicted = prediction.cpu().numpy()
            truth = target.cpu().numpy()
            for offset in range(predicted.shape[0]):
                hit = predicted[offset, 0] > 0
                real = truth[offset, 0] > 0
                per_image.append((len(per_image), int((hit & real).sum()),
                                  int((hit & ~real).sum()), int((~hit & real).sum())))
        if operating_point is not None:
            operating_point.update(outputs[0].sigmoid().cpu().numpy(), target.cpu().numpy())
        if spatial is not None:
            spatial.update(outputs[0] > 0, target)
    result = (sum(losses) / len(losses), meter.get_scores())
    return (*result, spatial.scores()) if spatial is not None else result


@torch.no_grad()
def evaluate_auxiliary(model, loader, device):
    """Measure guide and retrieval BCE separately from selection F1."""
    model.eval()
    sums = dict(retrieval_bce=0.)
    samples = 0
    for image, target in loader:
        image, target = image.to(device).float(), target.to(device).float()
        if model.module in ("difference", "manifold"):
            tar, difference = model.tar(model.encoder(image[:, :3])[1:], model.encoder(image[:, 3:])[1:])
            _, auxiliary = model.context(tar, difference, return_aux=True)
        else:
            tar = model.tar(model.encoder(image[:, :3])[1:], model.encoder(image[:, 3:])[1:])
            if model.module in ("hyperbolic", "matched"):
                _, auxiliary = model.context(tar, return_aux=True)
            else:
                _, auxiliary = model.context(tar)
        count = len(image)
        for name, value in guide_terms(auxiliary, target).items():
            metric = name + "_bce"
            sums[metric] = sums.get(metric, 0.) + float(value) * count
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


def save_checkpoint(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


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
    parser.add_argument("--module", choices=["none", "hyperbolic", "qkv", "matched", "difference", "manifold"],
                        required=True)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--encoder-lr-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--context-seed", type=int, default=2333)
    parser.add_argument("--context-heads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--detail-head", action="store_true")
    parser.add_argument("--detail-target", choices=("area", "nearest-exact"), default="area")
    parser.add_argument("--support", choices=("mass", "cell", "pixel", "none"), default="cell",
                        help="class prior added as log(support) to the context logits; "
                             "cell is the historical alias of mass")
    parser.add_argument("--kv-hierarchy", choices=("flat", "nested"), default="flat",
                        help="flat: 8/4/4/4 grids (224 KV); nested: 8/4/2/1 with parent-to-child pooling (170 KV)")
    parser.add_argument("--freeze-tau", action="store_true",
                        help="freeze the context temperature at the measured D0 median (0.4696)")
    parser.add_argument("--per-image-metrics", action="store_true",
                        help="store test-time per-image TP/FP/FN in per_image_cm.npz")
    parser.add_argument("--interior-positive-weight", type=float, default=1.0)
    parser.add_argument("--guide-weight", type=float)
    parser.add_argument("--retrieval-weight", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tar-variant", choices=("reparam", "plain", "diff", "spp", "spp1", "cg"),
                        default="reparam")
    parser.add_argument("--decoder-variant", choices=("reparam", "plain", "none"), default="reparam")
    parser.add_argument("--selection-split", choices=("test", "val"), default="test")
    parser.add_argument("--edge-weight", type=float, default=0.0,
                        help="weight of the Sobel edge loss read from the bridge or the retrieval")
    parser.add_argument("--edge-detach", choices=("none", "retrieval"), default="retrieval",
                        help="cut the edge gradient at the retrieval write-back")
    parser.add_argument("--edge-scales", choices=("all", "fine"), default="all",
                        help="which edge maps take the Sobel target: all four or the two finest")
    parser.add_argument("--aggregate", choices=("lorentz", "euclidean"), default="lorentz",
                        help="retrieval aggregation: Klein barycenter or a plain weighted mean")
    parser.add_argument("--edge-dilate", type=int, default=1,
                        help="pixels by which the GT edge ring is widened before supervision")
    args = parser.parse_args()
    if args.interior_positive_weight < 1:
        parser.error("interior positive weight must be at least 1")
    if args.edge_weight < 0:
        parser.error("edge weight must be non-negative")
    if args.edge_weight > 0 and args.tar_variant not in ("cg",) and args.module not in ("manifold", "difference"):
        parser.error("edge supervision needs an edge head: use --tar-variant cg or --module manifold")
    weights = auxiliary_weights(args.module, args.guide_weight, args.retrieval_weight)
    if weights["guide"] <= 0 or weights["geometric_retrieval"] < 0:
        parser.error("guide weight must be positive and retrieval weight nonnegative")
    if args.module not in ("hyperbolic", "matched", "difference", "manifold") and weights["geometric_retrieval"]:
        parser.error("retrieval supervision requires the matched H/E or difference context")
    if args.context_heads != 1 and args.module not in ("hyperbolic", "matched", "difference", "manifold"):
        parser.error("multihead ablation requires the matched H/E or difference context")
    if args.module in ("difference", "manifold") and args.tar_variant not in ("spp", "spp1", "diff", "cg"):
        parser.error("the difference context needs --tar-variant spp")
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
    val_loader = make_loader(args.dataset_root, "val", args.batch_size, args.num_workers)
    select_loader = test_loader if args.selection_split == "test" else val_loader
    epochs = math.ceil(args.max_steps / len(train_loader))
    actual_steps = epochs * len(train_loader)
    for index, loader in enumerate((train_loader, test_loader, val_loader)):
        generator = torch.Generator().manual_seed(args.seed + (0 if index == 0 else 1))
        loader.generator = generator
        if index == 0:
            loader.sampler.generator = generator
    model = CASANet(args.pretrained, module=args.module, context_seed=args.context_seed,
                    context_heads=args.context_heads, detail_head=args.detail_head,
                    detail_target=args.detail_target, tar_variant=args.tar_variant,
                    decoder_variant=args.decoder_variant, support=args.support,
                    kv_hierarchy=args.kv_hierarchy, freeze_tau=args.freeze_tau,
                    edge_supervision=args.edge_weight > 0,
                    edge_detach=args.edge_detach,
        prototype_pooling=args.prototype_pooling, aggregate_mode=args.aggregate).to(device)
    optimizer = make_optimizer(model, encoder_lr_mult=args.encoder_lr_mult)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    identity = source_files()
    manifest = dict(protocol=PROTOCOL, module=args.module, seed=args.seed,
                    tar_variant=args.tar_variant, decoder_variant=args.decoder_variant,
                    context_seed=args.context_seed,
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
                    selection=f"best {args.selection_split} F1 at threshold 0.5 each epoch, ties update",
                    selection_split=args.selection_split,
                    independent_test=args.selection_split == "val", evaluation_starts_at_epoch=0,
                    detail_head=args.detail_head, detail_supervision_weight=1. if args.detail_head else 0.,
                    detail_target=args.detail_target if args.detail_head else None,
                    context_heads=args.context_heads, test_primary_threshold=.5, calibration_split="val",
                    validation_threshold_grid=[float(f"{x:.3f}") for x in THRESHOLDS],
                    preprocessing=DATA_PROTOCOL, deep_supervision=4,
                    label_threshold=128, context_grid_sizes=list(model.context.grid_sizes),
                    context_prototypes_per_scale=list(model.context.tokens_per_scale),
                    joint_contexts=sum(model.context.tokens_per_scale),
                    qk_geometry=model.context.geometry, value_geometry=model.context.value_geometry,
                    qkv_dim=model.context.qk.out_channels if args.module != "none" else 0,
                    heads=model.context.heads if args.module != "none" else 0,
                    auxiliary_weights=weights,
                    edge_supervision=dict(weight=args.edge_weight, dilate=args.edge_dilate,
                                          detach=args.edge_detach,
                                          objective="sobel_edge_map_bce_plus_dice",
                                          source=("tar_cg_change_evidence" if args.tar_variant == "cg"
                                                  else "retrieval_writback")),
                    guide_objective=GUIDE_OBJECTIVE if args.module in ("hyperbolic", "matched", "difference", "manifold") else "raw_area_bce",
                    guide_effective_coefficients=guide_effective_coefficients(args.module, weights["guide"]),
                    query_key_head_dim=model.context.qk.out_channels // model.context.heads if args.module != "none" else 0,
                    value_head_dim=model.context.v.out_channels // model.context.heads if args.module != "none" else 0,
                    shared_query_key_projection=True,
                    compression_power=model.context.compression_power,
                    initial_compression_power=model.context.compression_power,
                    compression_reassignment=None,
                    interior_positive_weight=args.interior_positive_weight,
                    interior_weighting=dict(output=0, gt_boundary_radius=2,
                                           normalization="sum_weighted_bce/sum_pixel_weights"),
                    support_power=model.context.support_power,
                    support=args.support,
                    kv_hierarchy=args.kv_hierarchy,
                    freeze_tau=args.freeze_tau,
                    kv_tokens=(sum(2 * grid * grid for grid in model.context.grid_sizes)
                               if args.module in ("hyperbolic", "matched", "difference", "manifold") else 0),
                    self_reference=model.context.self_reference,
                    reliability_prior={"none": "not_used", "qkv": "weighted class posterior square",
                                       "hyperbolic": "weighted class occupancy; no entropy penalty",
                                       "matched": "weighted class occupancy; no entropy penalty",
                                       "difference": "temporal-difference class occupancy; no entropy penalty",
                                       "manifold": "temporal-difference class occupancy; no entropy penalty"}[args.module],
                    context_readout={"none": "identity", "qkv": "self-referenced conditional value difference",
                                     "hyperbolic": "learned gate on spatially centered retrieved context",
                                     "matched": "learned gate on spatially centered retrieved context",
                                     "difference": "learned gate on spatially centered retrieved context",
                                     "manifold": "lorentz barycenter with level path and position-aware gate"}[args.module],
                    hyperbolic_radius="asinh(norm(projected_features) / sqrt(head_dim))" if args.module in ("hyperbolic", "matched", "difference", "manifold") else None,
                    hyperbolic_key="lift_of_pooled_temporal_difference_content" if args.module in ("difference", "manifold")
                    else ("lift_of_shared_pooled_content" if args.module in ("hyperbolic", "matched") else None),
                    context_gamma_shape=[4, 64], local_gamma_shape=[2, 64],
                    local_scales=[1, 2], local_detail="depthwise3x3 + pointwise on X - avg3(X)",
                    source_files=identity,
                    warmup_steps=200, evaluation_split=args.selection_split,
                    precision=dict(input="float32", amp=False, cudnn_tf32=True,
                                   matmul_tf32=False, cudnn_benchmark=True),
                    context_values={"none": "guide_only",
                                    "qkv": "euclidean_relative_context",
                                    "hyperbolic": "direct_projection_class_keys_and_gated_euclidean_values",
                                    "matched": "direct_projection_class_keys_and_gated_euclidean_values",
                                    "difference": "class_keys_from_temporal_difference_and_gated_euclidean_values",
                                    "manifold": "class_keys_from_change_evidence_averaged_on_the_lorentz_barycenter"}[args.module],
                    source_archive=str(args.output_dir.parents[1] / "source_code.tar.gz"),
                    dataset=str(args.dataset_root), pretrained=args.pretrained,
                    deploy_verification=DEPLOY_POLICY)
    start_epoch, step, best_f1, best_epoch = 0, 0, -1.0, -1
    best_weights = None
    continuation = None
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for name in ("seed", "context_seed", "compression_reassignment", "auxiliary_weights", "guide_objective",
                     "interior_positive_weight", "context_heads", "detail_head", "detail_target",
                     "tar_variant", "decoder_variant",
                     "selection_split", "dataset", "batch_size", "validation_threshold_grid",
                     "support", "kv_hierarchy", "freeze_tau", "edge_supervision", "prototype_pooling"):
            if checkpoint["manifest"][name] != manifest[name]:
                raise RuntimeError(f"checkpoint training configuration differs: {name}")
        start_epoch, step, restored_rng = restore_training_state(
            checkpoint, model, optimizer, actual_steps, len(train_loader))
        best_f1, best_epoch = checkpoint["best_f1"], checkpoint["best_epoch"]
        best_weights = checkpoint["best_state_dict"]
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
        select_loader.generator.set_state(checkpoint["loader_rng_state"]["evaluation"])
        del checkpoint
    else:
        for parent in [args.output_dir, args.checkpoint_dir]:
            (parent / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (parent / "arch.json").write_text(json.dumps(dict(arch="casa", module=args.module,
                                                              str_dim=64, context_heads=args.context_heads,
                                                              tar_variant=args.tar_variant,
                                                              decoder_variant=args.decoder_variant,
                                                              selection_split=args.selection_split,
                                                              detail_head=args.detail_head,
                                                              support=args.support,
                                                              kv_hierarchy=args.kv_hierarchy,
                                                              freeze_tau=args.freeze_tau,
                                                              kv_tokens=manifest["kv_tokens"]), indent=2), encoding="utf-8")
    evaluation_manifest = dict(protocol=PROTOCOL, deploy_verification=DEPLOY_POLICY,
                               tolerance=2e-4, source_files=identity, continuation=continuation,
                               selection_split=args.selection_split,
                               independent_test=args.selection_split == "val", calibration_split="val",
                               test_backend="training settings restored after fold verification")
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2), encoding="utf-8")
    print(f"[PROTOCOL-VERSION] {PROTOCOL}", flush=True)
    print(f"[GUIDE-OBJECTIVE] {manifest['guide_objective']}", flush=True)
    print(f"[CONFIG] {json.dumps(manifest)}", flush=True)
    print(f"[TOTAL-PARAMS] {params:.6f} M", flush=True)
    print(f"[MAX-STEPS] requested={args.max_steps} actual={actual_steps} epochs={epochs} train_batches={len(train_loader)}", flush=True)
    session_start_step = step
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss, main_sum, auxiliary_sum, tick = 0.0, 0.0, 0.0, time.time()
        guide_sums, retrieval_sum = {}, 0.0
        for image, target in train_loader:
            image, target = image.to(device).float(), target.to(device).float()
            lr = learning_rate(5e-4, epoch, step, actual_steps)
            update_learning_rate(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            outputs, auxiliary = model(image[:, :3], image[:, 3:], return_aux=True)
            if not all(torch.isfinite(x).all() for x in outputs):
                raise FloatingPointError(f"non-finite prediction at step {step}")
            loss, main_loss, auxiliary_loss = training_losses(
                outputs, auxiliary, target, weights["guide"], weights["geometric_retrieval"],
                args.interior_positive_weight, args.edge_weight, args.edge_dilate,
                args.edge_scales)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if step % len(train_loader) == 0:
                with torch.no_grad():
                    print("[LOSS-DIAG] " + json.dumps(dict(
                        epoch=epoch, interior_positive_weight=args.interior_positive_weight,
                        interior_foreground_fraction=float((target * (mask_boundary(target, 2) == 0)).mean()),
                        main_unweighted=float(loss_function(outputs, target)),
                        detail_supervision_weight=manifest['detail_supervision_weight'],
                        main_optimized=float(main_loss))), flush=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                            error_if_nonfinite=True, foreach=True)
            optimizer.step()
            epoch_loss += float(loss.detach())
            main_sum += float(main_loss.detach())
            auxiliary_sum += float(auxiliary_loss.detach())
            with torch.no_grad():
                for name, value in guide_terms(auxiliary, target).items():
                    guide_sums[name] = guide_sums.get(name, 0.) + float(value)
                if "geometric_logits" in auxiliary:
                    retrieval_sum += float(guide_loss(auxiliary["geometric_logits"], target))
            step += 1
            if step % 100 == 0:
                print(f"[TRAIN] step={step}/{actual_steps} loss={float(loss.detach()):.5f} lr={lr:.8f} encoder_lr={lr*args.encoder_lr_mult:.8f}", flush=True)
        print(f"[ACTUAL-OPT-STEPS] epoch={epoch} executed={len(train_loader)} cumulative={step}/{actual_steps}", flush=True)
        guide_log = " ".join(f"{name}={value/len(train_loader):.6f}" for name, value in guide_sums.items())
        print(f"[TRAIN-EPOCH] epoch={epoch} loss={epoch_loss/len(train_loader):.6f} main={main_sum/len(train_loader):.6f} auxiliary={auxiliary_sum/len(train_loader):.6f} {guide_log} retrieval={retrieval_sum/len(train_loader):.6f} lr={lr:.8f} encoder_lr={lr*args.encoder_lr_mult:.8f}", flush=True)
        model.context.collect_diag = True
        eval_loss, score, spatial = evaluate(model, select_loader, device, return_diagnostics=True)
        model.context.collect_diag = False
        print(f"Epoch {epoch}/{epochs} | Loss={eval_loss:.6f} | Recall={score['recall']:.6f} | Precision={score['precision']:.6f} | OA={score['OA']:.6f} | F1={score['F1']:.6f} | IoU={score['IoU']:.6f} | Kappa={score['Kappa']:.6f} | t={time.time()-tick:.1f}s", flush=True)
        if model.context.geometry != "none":
            print(f"[MODULE-DIAG] {json.dumps(model.diagnostics())}", flush=True)
        print(f"[SPATIAL-DIAG] {json.dumps(dict(epoch=epoch, split=args.selection_split, **spatial))}", flush=True)
        improved = score["F1"] >= best_f1
        if improved:
            best_f1, best_epoch = float(score["F1"]), epoch
            best_weights = {name: tensor.detach().to('cpu', copy=True) for name, tensor in model.state_dict().items()}
        save_checkpoint(dict(epoch=epoch + 1, step=step, state_dict=model.state_dict(),
                             optimizer=optimizer.state_dict(), best_f1=best_f1, best_epoch=best_epoch,
                             best_state_dict=best_weights, manifest=manifest, rng_state=rng_state(),
                             loader_rng_state=dict(train=train_loader.generator.get_state(),
                                                   evaluation=test_loader.generator.get_state())), checkpoint_path)
        if improved:
            save_checkpoint(best_weights, args.checkpoint_dir / "best_model.pth")
    val_loader.generator = torch.Generator().manual_seed(args.seed + 1)
    last_auxiliary = evaluate_auxiliary(model, val_loader, device)
    _, last_val_score = evaluate(model, val_loader, device)
    save_checkpoint(best_weights, args.checkpoint_dir / "best_model.pth")
    model.load_state_dict(best_weights)
    best_auxiliary = evaluate_auxiliary(model, val_loader, device)
    (args.output_dir / "auxiliary_validation.json").write_text(json.dumps(dict(
        best_epoch=best_epoch, last_epoch=epochs-1, best=best_auxiliary, last=last_auxiliary,
        auxiliary_weights=weights,
        edge_supervision=manifest["edge_supervision"],
        guide_objective=manifest["guide_objective"],
        guide_effective_coefficients=manifest["guide_effective_coefficients"]), indent=2), encoding="utf-8")
    image, _ = next(iter(val_loader))
    image = image[:1].to(device).float()
    deploy, deployment = verify_deployment(model, image)
    difference, disagreement = deployment["max_diff"], deployment["binary_disagreement"]
    (args.output_dir / "deploy_check.json").write_text(json.dumps(deployment, indent=2), encoding="utf-8")
    operating_point = ThresholdCounts()
    evaluate(deploy, val_loader, device, operating_point=operating_point)
    calibration = operating_point.summary()
    threshold = calibration['best']['threshold']
    test_loader.generator = torch.Generator().manual_seed(args.seed+2)
    secondary = ThresholdCounts([threshold])
    per_image_rows = []
    test_loss, score, spatial = evaluate(
        deploy, test_loader, device, return_diagnostics=True, operating_point=secondary,
        per_image=per_image_rows if args.per_image_metrics else None)
    if args.per_image_metrics:
        table = np.asarray(per_image_rows, dtype=np.int64)
        list_path = Path(args.dataset_root) / "list" / "test.txt"
        list_lines = list_path.read_text(encoding="utf-8").splitlines() if list_path.is_file() else []
        np.savez(args.output_dir / "per_image_cm.npz", index=table[:, 0],
                 tp=table[:, 1], fp=table[:, 2], fn=table[:, 3],
                 # Provenance: the per-image rows are positional, so the split,
                 # threshold and list hash have to travel with them.
                 dataset=str(Path(args.dataset_root).name), split="test",
                 batch_size=args.batch_size, threshold=.5, images=len(table),
                 list_entries=len(list_lines), list_path=str(list_path),
                 list_sha256=hashlib.sha256("\n".join(list_lines).encode("utf-8")).hexdigest(),
                 protocol=PROTOCOL, seed=args.seed, selection_split=args.selection_split)
        print(f"[PER-IMAGE-CM] {len(table)} images -> per_image_cm.npz "
              f"(list {len(list_lines)} entries)", flush=True)
    (args.output_dir / "operating_point.json").write_text(json.dumps(dict(
        selection_split=args.selection_split, calibration_split="val", best_epoch=best_epoch,
        validation=calibration, test_calibrated=secondary.curve()[0]), indent=2), encoding="utf-8")
    deploy_params = sum(p.numel() for p in deploy.parameters()) / 1e6
    result = dict(protocol=PROTOCOL, module=args.module, metrics=score, best_epoch=best_epoch,
                  selection_split=args.selection_split,
                  independent_test=args.selection_split == "val", best_selection_f1=best_f1,
                  last_val_f1=last_val_score['F1'], calibration_split="val",
                  test_threshold=.5, test_calibrated=secondary.curve()[0],
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
