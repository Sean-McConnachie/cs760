import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # for lib

from tqdm import tqdm
import cv2
import argparse
from pathlib import Path

from lib import iter_dir_for_video_and_mask


def apply_mask_video(video_path: str, mask_path: str, out_path: str) -> None:
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    mask = cv2.VideoCapture(mask_path)
    if not mask.isOpened():
        raise RuntimeError(f"Could not open input mask: {mask_path}")

    fps = vid.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # XVID codec = AVI, works well in Windows
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    last_mframe = None
    while True:
        ret, vframe = vid.read()
        if not ret:
            break
        ret, mframe = mask.read()
        if not ret:
            if last_mframe is None:
                raise RuntimeError("Mask video is shorter than input video")
            mframe = last_mframe
        frame = cv2.bitwise_and(vframe, mframe)
        writer.write(frame)

    vid.release()
    writer.release()
    print(f"[DONE] Inverted mask saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Apply mask video (AND operation) and export as AVI")
    parser.add_argument("root_dir", help="Input root dir")
    parser.add_argument("output_dir", help="Output masked video dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # run once to ensure video and mask match and the program doesn't crash half way through
    for video in tqdm(iter_dir_for_video_and_mask(args.root_dir), desc="Checking video and mask"):
        out_fp = output_dir / f"{video['video_name']}.avi"
        apply_mask_video(video["video"], video["mask"], str(out_fp))

if __name__ == "__main__":
    main()
