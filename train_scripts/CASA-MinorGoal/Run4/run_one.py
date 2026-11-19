"""Launch one Run4 job: ``run_one.py <arm> <dataset> --gpu N [--resume]``."""
import argparse
import os

import config as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=C.ARM_ORDER)
    parser.add_argument("dataset", choices=C.DATASETS)
    parser.add_argument("--gpu", type=int, default=C.GPUS[0])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.arm, args.dataset) not in C.experiments():
        raise SystemExit(f"{(args.arm, args.dataset)} is not in the Run4 job matrix")

    out = C.run_dir(args.arm, args.dataset)
    ckpt = C.checkpoint_dir(args.arm, args.dataset)
    for path in (out, ckpt):
        path.mkdir(parents=True, exist_ok=True)

    command = [
        os.sys.executable, str(C.train_script()),
        "--dataset-root", str(C.DATA / args.dataset),
        "--output-dir", str(out),
        "--checkpoint-dir", str(ckpt),
        "--pretrained", str(C.PRETRAINED),
        *C.model_options(args.arm),
        "--seed", str(C.seed_for(args.arm)),
        "--context-seed", str(C.seed_for(args.arm)),
        "--batch-size", "32",
        "--max-steps", "40000",
        "--num-workers", "1",
        "--per-image-metrics",
    ]
    if args.resume and (ckpt / "last.pth").exists():
        command.append("--resume")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Appending on resume keeps the epoch history of the interrupted session.
    log = open(out / "train_log.txt", "a" if "--resume" in command else "w")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
