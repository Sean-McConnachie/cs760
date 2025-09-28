import cv2
import numpy as np
import os
import csv

orig_dir = "out_pairs/originals"
res_dir  = "outputs/StableDiffusion"
out_csv  = "fvd_scores.csv"

def load_video_frames(video_path, max_frames=16, resize=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0 and len(frames) < max_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame.astype(np.float32) / 255.0)
        count += 1
    cap.release()
    return np.array(frames)

def extract_features(frames):
    return frames.reshape(frames.shape[0], -1)

def simple_fvd(feat1, feat2):
    mu1, mu2 = feat1.mean(axis=0), feat2.mean(axis=0)
    sigma1, sigma2 = np.cov(feat1, rowvar=False), np.cov(feat2, rowvar=False)

    mean_diff = np.sum((mu1 - mu2)**2)
    cov_diff  = np.linalg.norm(sigma1 - sigma2, ord='fro')  # Frobenius norm

    return mean_diff + cov_diff




results = []

# Loop through originals and find matching results
for root, _, files in os.walk(orig_dir):
    for fname in files:
        if not fname.lower().endswith(".avi"):
            continue

        base, _ = os.path.splitext(fname)
        orig_path = os.path.join(root, fname)

        # Expected result filename pattern: "<base>_output.mp4"
        # res_fname = f"{base}_output.mp4"
        res_fname = f"{base}.avi"

        # If results are flat in res_dir, look there; otherwise try mirroring subfolders
        candidate_paths = [
            os.path.join(res_dir, res_fname),
            os.path.join(res_dir, base, res_fname),  # in case each result is in its own subfolder
        ]
        inp_path = next((p for p in candidate_paths if os.path.isfile(p)), None)

        if inp_path is None:
            print(f"[SKIP] No matching result for: {base}")
            continue

        # Load, feature-ize, score
        orig_frames = load_video_frames(orig_path)
        inp_frames  = load_video_frames(inp_path)

        if orig_frames.size == 0 or inp_frames.size == 0:
            print(f"[SKIP] Empty frames for: {base}")
            continue

        orig_feat = extract_features(orig_frames)
        inp_feat  = extract_features(inp_frames)
        try:
            score = simple_fvd(orig_feat, inp_feat)
        except Exception as e:
            print(f"[SKIP] Error computing FVD for {base}: {e}")
            continue

        results.append((base, score))
        print(f"[OK] {base}: {score}")

# Save CSV
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["video_name", "fvd_score"])
    for name, score in results:
        writer.writerow([name, score])

# # Paths
# orig_video = r"C:\Users\pc\Documents\Uni\Csc760\cs760\Diffueraser\dataset\originals\#122_Cleaning_Up_The_Beach_In_Chiba__Japan_pick_f_nm_np1_le_bad_2.avi"
# inp_video  = r"C:\Users\pc\Documents\Uni\Csc760\cs760\Diffueraser\results\#122_Cleaning_Up_The_Beach_In_Chiba__Japan_pick_f_nm_np1_le_bad_2_output.mp4"


# # Load and compute
# orig_feat = extract_features(load_video_frames(orig_video))
# inp_feat  = extract_features(load_video_frames(inp_video))

# fvd_score = simple_fvd(orig_feat, inp_feat)
# print("FVD score:", fvd_score)
print(f"\nSaved {len(results)} rows to {out_csv}")