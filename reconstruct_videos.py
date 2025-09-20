import cv2
import os

def frames_to_video(frames_folder, output_file, fps=30):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])
    if not frame_files:
        print("No frames found in", frames_folder)
        return

    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for f in frame_files:
        frame = cv2.imread(os.path.join(frames_folder, f))
        out.write(frame)

    out.release()
    print(f"Video saved to {output_file}")

orig_frames_folder = r"C:\Users\Sreelakshmi Gireesh\video_inpainting_dataset\inpainting_full\inpainting\inpainting\src\mallard-fly"
inp_frames_folder  = r"C:\Users\Sreelakshmi Gireesh\video_inpainting_dataset\inpainting_full\inpainting\inpainting\fake\DFGVI\mallard-fly"

orig_video_file = r"C:\Users\Sreelakshmi Gireesh\video_inpainting_dataset\mallard-fly_orig.mp4"
inp_video_file  = r"C:\Users\Sreelakshmi Gireesh\video_inpainting_dataset\mallard-fly_inp.mp4"

frames_to_video(orig_frames_folder, orig_video_file)
frames_to_video(inp_frames_folder, inp_video_file)
