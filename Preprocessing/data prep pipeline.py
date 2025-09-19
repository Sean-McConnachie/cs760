# build_and_sample_hmdb51.py
# ------------------------------------------------------------
# 1) Build a full manifest from HMDB51 (AVI)
# 2) Detect letterbox per clip (has_letterbox, letterbox_ratio)
# 3) Sample 500 clips with the tiered policy
# ------------------------------------------------------------

import os
import cv2
import numpy as np
import pandas as pd

# ====================== USER CONFIG =========================
DATASET_DIR   = r"C:\Users\jamie\Desktop\760 research\Plan b\HMDB51"
MANIFEST_CSV  = "hmdb51_manifest.csv"
SAMPLED_CSV   = "hmdb51_sample500_pref_noletterbox.csv"

TARGET_N      = 500
SEED          = 760

# Hard constraints
REQ_WIDTH     = 320
REQ_HEIGHT    = 240
DUR_MIN_HARD  = 2.0
DUR_MAX_HARD  = 6.0
REQ_CM        = "static"  # camera_motion must equal this (parsed from filename)

# Tier thresholds (letterbox thickness as fraction of short side)
TIER2_MAX_RATIO = 0.03    # <= 3% ~ near no-letterbox
TIER3_MAX_RATIO = 0.05    # <= 5% ~ thin letterbox

# Final fallback if still < 500 after Tier3
DUR_MIN_SOFT  = 1.8
DUR_MAX_SOFT  = 6.5
# ============================================================


# ------------------- Letterbox detection --------------------
def _first_non_black_index(values, thr=10):
    """Return first index where row/col mean > thr; else len(values)."""
    idx = np.argmax(values > thr)
    return int(idx) if values.size and values[idx] > thr else int(values.size)

def _last_non_black_suffix(values, thr=10):
    """Return count of trailing rows/cols <= thr; else 0."""
    inv = values[::-1]
    idx = np.argmax(inv > thr)
    return int(idx) if inv.size and inv[idx] > thr else int(inv.size)

def _estimate_borders_from_frame(frame_bgr, thr=10):
    """Estimate top/bottom/left/right black borders (in pixels) from a frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Per-row/col mean intensity
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)

    # contiguous black runs from edges
    top    = _first_non_black_index(row_mean, thr=thr)
    bot    = _last_non_black_suffix(row_mean, thr=thr)
    left   = _first_non_black_index(col_mean, thr=thr)
    right  = _last_non_black_suffix(col_mean, thr=thr)

    # Cap to image bounds (robustness)
    top    = int(np.clip(top,  0, h))
    bot    = int(np.clip(bot,  0, h - top))
    left   = int(np.clip(left, 0, w))
    right  = int(np.clip(right,0, w - left))
    return top, bot, left, right, (h, w)

def detect_letterbox(video_path, sample_positions=(0.1, 0.5, 0.9), thr=10):
    """
    Detect letterbox borders by sampling a few frames.
    Returns:
      has_letterbox (bool),
      letterbox_ratio (float in [0,1], thickness fraction of short side),
      borders_px = dict(top, bottom, left, right) in pixels (median over samples)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return True, 1.0, {"top":0,"bottom":0,"left":0,"right":0}  # treat unknown as letterboxed

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count <= 0:
        cap.release()
        return True, 1.0, {"top":0,"bottom":0,"left":0,"right":0}

    tops, bots, lefts, rights = [], [], [], []
    H, W = None, None

    for p in sample_positions:
        idx = int(max(0, min(frame_count - 1, round(p * (frame_count - 1)))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        t, b, l, r, (h, w) = _estimate_borders_from_frame(frame, thr=thr)
        tops.append(t); bots.append(b); lefts.append(l); rights.append(r)
        H, W = h, w

    cap.release()

    if not tops:
        return True, 1.0, {"top":0,"bottom":0,"left":0,"right":0}

    # Median over sampled frames for robustness
    top = int(np.median(tops)); bottom = int(np.median(bots))
    left = int(np.median(lefts)); right = int(np.median(rights))

    # Letterbox thickness ratios along vertical/horizontal
    # (sum of opposite borders over that dimension)
    v_ratio = (top + bottom) / max(H, 1)
    h_ratio = (left + right) / max(W, 1)

    # Define a single scalar ratio as the worst-case fraction of the SHORT side
    short_side = min(H, W)
    # take the max thickness among top/bottom/left/right
    max_thickness_px = max(top, bottom, left, right)
    letterbox_ratio = max_thickness_px / max(short_side, 1)

    # has_letterbox if any meaningful border present (> ~0.5% of short side or >=2 px)
    has_letterbox = (letterbox_ratio > 0.005) or (max_thickness_px >= 2)

    return bool(has_letterbox), float(letterbox_ratio), {
        "top": top, "bottom": bottom, "left": left, "right": right
    }


# ------------------- Metadata parsing -----------------------
def parse_from_filename(filename_noext):
    """
    Parse camera_motion and quality from HMDB51-style filename.
    Examples:
      *_cm_* => motion; *_nm_* => static
      *_goo => good; *_med => medium; *_bad => bad
    """
    name = filename_noext.lower()
    camera_motion = "motion" if "_cm_" in name else "static"  # default static if not found
    if "_goo" in name:
        quality = "good"
    elif "_med" in name:
        quality = "medium"
    elif "_bad" in name:
        quality = "bad"
    else:
        quality = "unknown"
    return camera_motion, quality


# ------------------- Manifest building ----------------------
def probe_video_props(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    duration = frames / fps if fps > 0 else 0.0
    cap.release()
    return dict(fps=fps, frames=frames, width=width, height=height, duration_sec=duration)

def build_manifest(dataset_dir, out_csv):
    rows = []
    exts = (".avi", ".mp4")
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if not f.lower().endswith(exts):
                continue
            file_path = os.path.join(root, f)
            filename_noext = os.path.splitext(f)[0]
            props = probe_video_props(file_path)
            if props is None:
                continue

            # Parse labels
            class_name = os.path.basename(root)  # folder = action class
            camera_motion, quality = parse_from_filename(filename_noext)

            # Letterbox detection
            has_L, L_ratio, L_borders = detect_letterbox(file_path)

            rows.append({
                "clip_id": filename_noext,
                "path": os.path.abspath(file_path),
                "class": class_name,
                "camera_motion": camera_motion,
                "video_quality": quality,
                "fps": props["fps"],
                "frames": props["frames"],
                "width": props["width"],
                "height": props["height"],
                "duration_sec": props["duration_sec"],
                "has_letterbox": has_L,
                "letterbox_ratio": L_ratio,
                "L_top_px": L_borders["top"],
                "L_bottom_px": L_borders["bottom"],
                "L_left_px": L_borders["left"],
                "L_right_px": L_borders["right"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Manifest saved: {out_csv} (rows={len(df)})")
    return df


# ------------------- Sampling policy ------------------------
def sample_df(df: pd.DataFrame, n: int, seed: int):
    if len(df) <= 0:
        return df.iloc[0:0].copy()
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed)

def run_sampling(df: pd.DataFrame) -> pd.DataFrame:
    # Hard filters
    base = df[
        (df["width"] == REQ_WIDTH) &
        (df["height"] == REQ_HEIGHT) &
        (df["duration_sec"] >= DUR_MIN_HARD) &
        (df["duration_sec"] <= DUR_MAX_HARD) &
        (df["camera_motion"].str.lower() == REQ_CM)
    ].copy()

    print(f"[INFO] After hard filters: {len(base)} rows")

    selected_parts = []
    remaining = TARGET_N

    # Tier 1: no letterbox
    t1 = base[base["has_letterbox"] == False]
    print(f"[Tier1] no-letterbox: {len(t1)}")
    take1 = sample_df(t1, remaining, SEED)
    selected_parts.append(take1)
    remaining -= len(take1)

    if remaining > 0:
        # Tier 2: very thin letterbox (<= 3%)
        leftover = base.drop(index=take1.index)
        t2 = leftover[(leftover["has_letterbox"] == True) &
                      (leftover["letterbox_ratio"] <= TIER2_MAX_RATIO)]
        print(f"[Tier2] letterbox_ratio <= {TIER2_MAX_RATIO:.2f}: {len(t2)}")
        take2 = sample_df(t2, remaining, SEED + 1)
        selected_parts.append(take2)
        remaining -= len(take2)

    if remaining > 0:
        # Tier 3: thin letterbox (<= 5%)
        leftover = base.drop(index=pd.concat(selected_parts).index.unique())
        t3 = leftover[(leftover["has_letterbox"] == True) &
                      (leftover["letterbox_ratio"] <= TIER3_MAX_RATIO)]
        print(f"[Tier3] letterbox_ratio <= {TIER3_MAX_RATIO:.2f}: {len(t3)}")
        take3 = sample_df(t3, remaining, SEED + 2)
        selected_parts.append(take3)
        remaining -= len(take3)

    if remaining > 0:
        # Final fallback: relax duration to 1.8–6.5s (still 320x240, static)
        relaxed = df[
            (df["width"] == REQ_WIDTH) &
            (df["height"] == REQ_HEIGHT) &
            (df["duration_sec"] >= DUR_MIN_SOFT) &
            (df["duration_sec"] <= DUR_MAX_SOFT) &
            (df["camera_motion"].str.lower() == REQ_CM)
        ].copy()
        already = pd.concat(selected_parts).index.unique() if selected_parts else []
        relaxed = relaxed.drop(index=already, errors="ignore")
        print(f"[Fallback] duration 1.8–6.5s: {len(relaxed)}")
        takeF = sample_df(relaxed, remaining, SEED + 3)
        selected_parts.append(takeF)
        remaining -= len(takeF)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else base.iloc[0:0].copy()
    print(f"[INFO] Selected total = {len(selected)} (target={TARGET_N})")
    return selected


# =========================== MAIN ===========================
if __name__ == "__main__":
    # Step 1: build manifest
    if not os.path.exists(MANIFEST_CSV):
        print("[INFO] Building manifest…")
        manifest = build_manifest(DATASET_DIR, MANIFEST_CSV)
    else:
        print(f"[INFO] Loading existing manifest: {MANIFEST_CSV}")
        manifest = pd.read_csv(MANIFEST_CSV)

    # Basic sanity
    required_cols = ["width","height","duration_sec","camera_motion","has_letterbox","letterbox_ratio"]
    missing = [c for c in required_cols if c not in manifest.columns]
    if missing:
        raise RuntimeError(f"Manifest missing columns: {missing}")

    # Step 2: run sampling policy
    sampled = run_sampling(manifest)

    # Optional: sort by clip_id for determinism
    if "clip_id" in sampled.columns:
        sampled = sampled.sort_values("clip_id").reset_index(drop=True)

    # Step 3: save
    sampled.to_csv(SAMPLED_CSV, index=False)
    print(f"Saved sampled manifest: {SAMPLED_CSV}")
