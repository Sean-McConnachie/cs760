import cv2
import argparse
from pathlib import Path

def invert_mask_video(in_path: str, out_path: str) -> None:
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input mask: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # XVID codec = AVI, works well in Windows
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inverted = 255 - frame
        writer.write(inverted)

    cap.release()
    writer.release()
    print(f"[DONE] Inverted mask saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Invert mask video (black <-> white) and export as AVI")
    parser.add_argument("input", help="Input mask video file (any format)")
    parser.add_argument("output", help="Output inverted mask video (AVI)")
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.suffix.lower() != ".avi":
        raise ValueError("Output file must end with .avi")

    invert_mask_video(args.input, str(out_path))

if __name__ == "__main__":
    main()
