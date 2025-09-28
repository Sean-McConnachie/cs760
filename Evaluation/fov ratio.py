import os
import cv2
import numpy as np
import pandas as pd  


orig_base = "frames/original"
inp_base = "frames/inpainted"  

video_names = sorted(os.listdir(orig_base))

fov_results = {}

for video_name in video_names:
    orig_folder = os.path.join(orig_base, video_name)
    # inp_folder = os.path.join(inp_base, video_name + "_output") # Adjusted to match the output folder naming convention
    inp_folder = os.path.join(inp_base, video_name)
    inp_frames = []
    orig_frames = sorted(os.listdir(orig_folder))
    if(os.path.isdir(inp_folder)==False):
        print(f"⚠️ Skipping {video_name}, no inpainted folder found.")
        continue
    else:
        inp_frames = sorted(os.listdir(inp_folder))
    
    fov_ratios = []
    
    for o_frame, i_frame in zip(orig_frames, inp_frames):
        orig_img = cv2.imread(os.path.join(orig_folder, o_frame))
        inp_img = cv2.imread(os.path.join(inp_folder, i_frame))
        
        if orig_img is None or inp_img is None:
            continue
        
        non_black = np.count_nonzero(cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY))
        total_pixels = inp_img.shape[0] * inp_img.shape[1]
        fov_ratios.append(non_black / total_pixels)
    
    avg_fov = np.mean(fov_ratios)
    fov_results[video_name] = avg_fov
    print(f"Average Preserved FOV Ratio for '{video_name}': {avg_fov:.4f}")
    
df = pd.DataFrame(list(fov_results.items()), columns=["Video_Name", "Avg_FOV_Ratio"])
df.to_csv("fov_results.csv", index=False)
print("\nResults saved to fov_results.csv")
