"""Launch one Run16 job: ``run_one.py <version> <arm> <dataset> --gpu N [--resume]``."""
import argparse
import os

import config as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("arm")
    parser.add_argument("dataset")
    parser.add_argument("--gpu", type=int, default=C.GPUS[0])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = C.run_dir(args.version, args.arm, args.dataset)
    ckpt = C.checkpoint_dir(args.version, args.arm, args.dataset)
    for path in (out, ckpt):
        path.mkdir(parents=True, exist_ok=True)

    script = C.train_script(args.version)
    command = [
        os.sys.executable, str(script),
        "--dataset-root", str(C.DATA / args.dataset),
        "--output-dir", str(out),
        "--checkpoint-dir", str(ckpt),
        "--pretrained", str(C.PRETRAINED),
        *C.model_options(args.version, args.arm),
        "--batch-size", "32",
        "--max-steps", "40000",
        "--num-workers", "1",
    ]
    if args.resume and (ckpt / "last.pth").exists():
        command.append("--resume")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    log = open(out / "train_log.txt", "w")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    print(f"[COMMAND] {' '.join(command)}", flush=True)
    print(f"[CUDA_VISIBLE_DEVICES] {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()

