"""Download Run3 artifacts and rebuild the attribution table."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("CASA_REPO", str(HERE.parents[2])))


def run(command):
    print("$", " ".join(str(item) for item in command), flush=True)
    subprocess.run([str(item) for item in command], check=True, cwd=REPO)


def main():
    local_root = REPO / "outputs/CASA-MinorGoal/Run3"
    local_root.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "tools/download_training_results.py",
         "--tag", "CASA-MinorGoal", "--run", "Run3"])
    run([sys.executable, "tools/extract_metrics_to_excel.py"])
    run([sys.executable, "tools/models_to_txt.py", "--tag", "CASA-MinorGoal", "--run", "Run3"])
    run([sys.executable, str(HERE / "compare.py"), "--root", str(local_root)])
    print(f"[COLLECTED] {local_root}")


if __name__ == "__main__":
    main()
