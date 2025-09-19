import os
import cv2
import numpy as np
import pandas as pd
import hashlib
import shutil

# ======= User settings =======
CSV_PATH          = "hmdb51_sample500_pref_noletterbox.csv"
OUTPUT_ROOT       = "out_pairs"                 # root folder for outputs
ORIG_DIR          = os.path.join(OUTPUT_ROOT, "originals")
MASK_DIR          = os.path.join(OUTPUT_ROOT, "masks")
INDEX_CSV         = os.path.join(OUTPUT_ROOT, "mask_pairs_index.csv")

OVERWRITE_MASK    = False   # set True to regenerate masks
OVERWRITE_ORIG    = False   # set True to recopy originals

# Viewport (crop window) & motion parameters
BASE_VISIBLE_RATIO = 0.85   # fraction of (W,H)
MAX_TX, MAX_TY     = 12, 12 # px translation
MAX_ROT_DEG        = 2.0    # ±degrees
MAX_ZOOM_DELTA     = 0.05   # scale
SMOOTH_WIN         = 21     # odd; larger => smoother low-frequency motion

# ======= Helpers =======
def seed_from_clip_id(clip_id: str, base_seed: int = 760) -> int:
    """Stable per-clip seed derived from clip_id (reproducible across machines)."""
    h = hashlib.md5(clip_id.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ base_seed) & 0x7FFFFFFF

def smooth_ma_reflect(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average with reflection padding to avoid start/end bias."""
    if k <= 1:
        return x
    k = int(k) | 1
    pad = k // 2
    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    ker = np.ones(k, dtype=np.float32) / k
    return np.convolve(x_pad, ker, mode="valid")

def generate_mask(video_path: str, out_mask_path: str, seed: int):
    """Create a binary (0/255) viewport mask video aligned to the input clip.
       Returns (fps, frames, width, height)."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc  = cv2.VideoWriter_fourcc(*"XVID")
    vw_mask = cv2.VideoWriter(out_mask_path, fourcc, fps, (W, H), isColor=False)
    assert vw_mask.isOpened(), f"Cannot open writer: {out_mask_path}"

    # Low-frequency random trajectories (deterministic per clip)
    rng = np.random.default_rng(seed)
    tx = rng.uniform(-MAX_TX, MAX_TX, size=N)
    ty = rng.uniform(-MAX_TY, MAX_TY, size=N)
    rt = rng.uniform(-MAX_ROT_DEG, MAX_ROT_DEG, size=N) * np.pi/180.0
    sc = 1.0 + rng.uniform(-MAX_ZOOM_DELTA, MAX_ZOOM_DELTA, size=N)
    tx, ty, rt, sc = map(lambda v: smooth_ma_reflect(v, SMOOTH_WIN), (tx, ty, rt, sc))

    # Centered viewport polygon
    bw = int(W * BASE_VISIBLE_RATIO)
    bh = int(H * BASE_VISIBLE_RATIO)
    base = np.array([[-bw/2, -bh/2],
                      [ bw/2, -bh/2],
                      [ bw/2,  bh/2],
                      [-bw/2,  bh/2]], dtype=np.float32)
    cx, cy = W/2.0, H/2.0

    written = 0
    for i in range(N):
        ok, _ = cap.read()
        if not ok:
            break
        c, s = np.cos(rt[i]) * sc[i], np.sin(rt[i]) * sc[i]
        R = np.array([[c, -s], [s,  c]], dtype=np.float32)
        pts = (base @ R.T)
        pts[:, 0] += cx + tx[i]
        pts[:, 1] += cy + ty[i]

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)  # crisp binary
        vw_mask.write(mask)
        written += 1

    cap.release()
    vw_mask.release()
    if written == 0:
        raise RuntimeError(f"No frames written for: {video_path}")

    return float(fps), int(N), int(W), int(H)

# ======= Main =======
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(ORIG_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    rows = []

    for row in df.itertuples(index=False):
        clip_id = getattr(row, "clip_id")
        src_path = getattr(row, "path")

        # Original output name: originals/{clip_id}.avi
        orig_ext = os.path.splitext(src_path)[1].lower() or ".avi"
        orig_out = os.path.join(ORIG_DIR, f"{clip_id}{orig_ext}")

        # Mask output name: masks/{clip_id}_mask.avi
        mask_out = os.path.join(MASK_DIR, f"{clip_id}_mask.avi")

        # Copy original (no re-encode, preserves GT bytes)
        if OVERWRITE_ORIG or not os.path.exists(orig_out):
            shutil.copy2(src_path, orig_out)

        # Generate mask
        if OVERWRITE_MASK or not os.path.exists(mask_out):
            seed = seed_from_clip_id(str(clip_id))
            fps, frames, W, H = generate_mask(src_path, mask_out, seed)
        else:
            # If mask already exists and we don't overwrite, read basic meta for index
            cap_tmp = cv2.VideoCapture(src_path)
            fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
            frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
            W = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_tmp.release()
            seed = seed_from_clip_id(str(clip_id))  # still record seed

        # Store relative paths in index for portability
        rows.append({
            "clip_id": clip_id,
            "original_rel": os.path.relpath(orig_out, OUTPUT_ROOT).replace("\\", "/"),
            "mask_rel": os.path.relpath(mask_out, OUTPUT_ROOT).replace("\\", "/"),
            "fps": fps,
            "frames": frames,
            "width": W,
            "height": H,
            "seed": seed
        })

    index_df = pd.DataFrame(rows)
    index_df.to_csv(INDEX_CSV, index=False)
    print(f"Done. Wrote {len(index_df)} pairs and index:\n  {INDEX_CSV}")
    print(f"   Originals dir: {ORIG_DIR}\n   Masks dir    : {MASK_DIR}")

if __name__ == "__main__":
    main()
