"""Download log and JSON artifacts for one remote experiment family/run."""
import argparse
import json
from pathlib import Path
import stat

import paramiko

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / ".vscode/sftp.json").read_text(encoding="utf-8"))
    remote = f"/home/yqwang/outputs/CASA-CD/{args.tag}/{args.run}"
    local = ROOT / "outputs" / args.tag / args.run
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(config["host"], config.get("port", 22), config["username"], config["password"], timeout=30)
    count = 0
    try:
        with client.open_sftp() as sftp:
            def visit(source, target):
                nonlocal count
                target.mkdir(parents=True, exist_ok=True)
                for entry in sftp.listdir_attr(source):
                    child = f"{source}/{entry.filename}"
                    if stat.S_ISDIR(entry.st_mode):
                        visit(child, target / entry.filename)
                    elif Path(entry.filename).suffix in (".txt", ".json", ".log", ".csv", ".md"):
                        sftp.get(child, str(target / entry.filename))
                        count += 1
            visit(remote, local)
    finally:
        client.close()
    print(f"Downloaded {count} artifacts to {local}")


if __name__ == "__main__":
    main()
