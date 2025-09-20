import os
import shutil

orig_videos_path = r"C:\Users\Sreelakshmi Gireesh\video_inpainting_dataset\inpainting_full\inpainting\inpainting\src"
inpainted_videos_path = r"C:\Users\Sreelakshmi Gireesh\video_inpainting_dataset\inpainting_full\inpainting\inpainting\fake"

orig_frames_out = "frames/original"
inp_frames_out = "frames/inpainted"

def copy_frames(video_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for folder_name in os.listdir(video_path):
        folder_path = os.path.join(video_path, folder_name)
        out_folder = os.path.join(output_path, folder_name)
        os.makedirs(out_folder, exist_ok=True)
        
        for file_name in os.listdir(folder_path):
            src_file = os.path.join(folder_path, file_name)
            dst_file = os.path.join(out_folder, file_name)
            shutil.copy(src_file, dst_file)

copy_frames(orig_videos_path, orig_frames_out)

copy_frames(os.path.join(inpainted_videos_path, "DFGVI"), os.path.join(inp_frames_out, "DFGVI"))

print("Original frames folders:", os.listdir(orig_frames_out))
print("Inpainted frames folders:", os.listdir(os.path.join(inp_frames_out, "DFGVI")))
