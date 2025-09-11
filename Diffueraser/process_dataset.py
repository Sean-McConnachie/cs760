import sys, shutil, tempfile, subprocess, argparse
from pathlib import Path
import cv2
from lib import iter_dir_for_video_and_mask

def normalize_mask(in_path: str, out_path: str, semantics: str) -> None:
    """
    semantics:
      - 'white_is_fill' (default): white (255) = inpaint area, black (0) = keep
      - 'black_is_fill': black (0) = inpaint area, will be inverted to white
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open mask: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*("mp4v"))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # convert to grayscale then hard-threshold -> binary 0/255
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binmask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        if semantics == "black_is_fill":
            binmask = 255 - binmask  # flip semantics: black fill -> white fill

        # Write 3-channel video (keep it simple)
        writer.write(cv2.cvtColor(binmask, cv2.COLOR_GRAY2BGR))

    cap.release(); writer.release()

def find_output_file(work_dir: Path) -> Path | None:
    # Prefer the common filename; else pick most recent video produced.
    for name in ("diffueraser_result.mp4", "diffueraser_result.avi", "priori.mp4"):
        p = work_dir / name
        if p.exists():
            return p
    vids = sorted((list(work_dir.glob("*.mp4")) + list(work_dir.glob("*.avi"))),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return vids[0] if vids else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="dataset\Videos")
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--video_suffix", default=".avi")
    ap.add_argument("--mask_suffix",  default="_mask.avi")
    ap.add_argument("--run_script",   default="run_diffueraser.py")
    ap.add_argument("--mask_semantics", choices=["white_is_fill", "black_is_fill"],
                    default="black_is_fill",
                    help="Set to black_is_fill if your masks use black to indicate the area to remove")
    args = ap.parse_args()

    in_dir  = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = iter_dir_for_video_and_mask(
        dir=str(in_dir),
        video_suffix=args.video_suffix,
        mask_suffix=args.mask_suffix,
    )
    print(f"[INFO] Found {len(pairs)} video+mask pairs")

    for pair in pairs:
        name  = pair["video_name"]
        video = pair["video"]
        mask  = pair["mask"]

        final_out = out_dir / f"{name}_output.mp4"
        if final_out.exists():
            print(f"[SKIP] {name} already processed -> {final_out.name}")
            continue

        work_dir = Path(tempfile.mkdtemp(prefix=f"diffueraser_{name}_"))
        try:
            # Normalize mask to expected semantics (white = fill)
            mask_norm = work_dir / "mask_norm.mp4"
            normalize_mask(mask, str(mask_norm), args.mask_semantics)

            # DiffuEraser expects --save_path as a DIRECTORY; we feed the temp dir.
            cmd = [
                sys.executable, args.run_script,
                "--input_video", video,
                "--input_mask", str(mask_norm),
                "--save_path", str(work_dir),
            ]
            print(f"[RUN] {name}")
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"[FAIL] {name} (rc={rc})")
                continue

            produced = find_output_file(work_dir)
            if not produced:
                print(f"[FAIL] {name} (no output found in {work_dir})")
                continue

            shutil.move(str(produced), str(final_out))
            print(f"[DONE] {name} -> {final_out.name}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
