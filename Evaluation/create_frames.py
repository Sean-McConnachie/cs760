import cv2
import os

orig_videos_path = f"out_pairs/originals"
inpainted_videos_path = f"outputs/wan_1.3"


orig_frames_out = "frames/original"
inp_frames_out = "frames/inpainted"

def extract_frames(video_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    for file_name in os.listdir(video_path):
        if file_name.endswith(".mp4") or file_name.endswith(".avi"):  # only process videos
            video_file = os.path.join(video_path, file_name)
            cap = cv2.VideoCapture(video_file)

            frame_idx = 0
            video_name = os.path.splitext(file_name)[0]
            out_folder = os.path.join(output_path, video_name)
            os.makedirs(out_folder, exist_ok=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_file = os.path.join(out_folder, f"frame{frame_idx:05d}.jpg")
                cv2.imwrite(frame_file, frame)
                frame_idx += 1

            cap.release()
            print(f"Extracted {frame_idx} frames from {file_name}")

extract_frames(orig_videos_path, orig_frames_out)
extract_frames(inpainted_videos_path, inp_frames_out)

print("Frame extraction completed.")