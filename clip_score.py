import os
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

orig_folder = r"C:\Users\Sreelakshmi Gireesh\video_inpainting_dataset\inpainting_full\inpainting\inpainting\src\mallard-fly"
inp_folder  = r"C:\Users\Sreelakshmi Gireesh\video_inpainting_dataset\inpainting_full\inpainting\inpainting\fake\DFGVI\mallard-fly"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)

def get_embeddings(frame_folder):
    """Compute CLIP embeddings for all frames in a folder."""
    frame_files = sorted(os.listdir(frame_folder))
    embeddings = []

    for f in tqdm(frame_files, desc="Processing frames"):
        img_path = os.path.join(frame_folder, f)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image)
            emb /= emb.norm(dim=-1, keepdim=True)  # normalize
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

print("Computing embeddings for original frames...")
orig_embs = get_embeddings(orig_folder)
print("Computing embeddings for inpainted frames...")
inp_embs = get_embeddings(inp_folder)

def frame_consistency(embs):
    sims = []
    for i in range(len(embs)-1):
        sim = cosine_similarity(embs[i:i+1], embs[i+1:i+2])[0][0]
        sims.append(sim)
    return np.mean(sims)

orig_score = frame_consistency(orig_embs)
inp_score  = frame_consistency(inp_embs)

print(f"Frame Consistency CLIP Score - Original: {orig_score:.4f}")
print(f"Frame Consistency CLIP Score - Inpainted: {inp_score:.4f}")
