import os
import sys
import subprocess
from pathlib import Path
from lib import iter_dir_for_video_and_mask  # your helper funcs

def main(
    input_dir: str = "dataset",
    output_dir: str = "results",
    video_suffix: str = ".avi",
    mask_suffix: str = "_shake_mask.avi",
    run_script: str = "run_diffueraser.py",
):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = iter_dir_for_video_and_mask(
        dir=str(input_dir),
        video_suffix=video_suffix,
        mask_suffix=mask_suffix,
    )

    print(f"[INFO] Found {len(pairs)} video+mask pairs")

    for pair in pairs:
        name = pair["video_name"]
        video = pair["video"]
        mask = pair["mask"]

        out_dir = output_dir / name
        if out_dir.exists():
            print(f"[SKIP] {name} already processed")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, run_script,
            "--input_video", video,
            "--input_mask", mask,
            "--out_dir", str(out_dir),
        ]

        print(f"[RUN] Processing {name} …")
        ret = subprocess.call(cmd)

        if ret == 0:
            print(f"[DONE] {name}")
        else:
            print(f"[FAIL] {name} (return code {ret})")

if __name__ == "__main__":
    main()
