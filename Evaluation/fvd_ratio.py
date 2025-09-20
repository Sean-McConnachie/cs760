import cv2
import numpy as np
import os

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

# Paths
orig_video = r"C:\Users\Sreelakshmi Gireesh\video_inpainting_dataset\mallard-fly_orig.mp4"
inp_video  = r"C:\Users\Sreelakshmi Gireesh\video_inpainting_dataset\mallard-fly_inp.mp4"

# Load and compute
orig_feat = extract_features(load_video_frames(orig_video))
inp_feat  = extract_features(load_video_frames(inp_video))

fvd_score = simple_fvd(orig_feat, inp_feat)
print("FVD score:", fvd_score)
