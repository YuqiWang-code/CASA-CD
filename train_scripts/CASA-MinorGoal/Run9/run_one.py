"""Launch one Run9 job: ``run_one.py <arm> <dataset> --gpu N [--resume]``."""
import argparse
import os

import config as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=tuple(C.ARMS))
    parser.add_argument("dataset", choices=tuple(C.DATASETS))
    parser.add_argument("--gpu", type=int, default=C.GPUS[0])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.arm, args.dataset) not in C.experiments():
        raise SystemExit(f"{(args.arm, args.dataset)} is not in the Run9 job matrix")

    out = C.run_dir(args.arm, args.dataset)
    ckpt = C.checkpoint_dir(args.arm, args.dataset)
    for path in (out, ckpt):
        path.mkdir(parents=True, exist_ok=True)

    command = [
        os.sys.executable, str(C.train_script()),
        "--dataset-root", str(C.DATA / args.dataset),
        "--output-dir", str(out),
        "--checkpoint-dir", str(ckpt),
        "--pretrained", str(C.pretrained_for(args.arm)),
        *C.model_options(args.arm),
        "--seed", str(C.seed_for(args.arm)),
        "--context-seed", str(C.seed_for(args.arm)),
        "--batch-size", str(C.batch_for(args.arm)),
        "--max-steps", str(C.steps_for(args.arm)),
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
