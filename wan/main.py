import os
import sys
COMFY_UI_REPO_PATH = "/home/dude-desktop/dev/cs760-OLD/ComfyUI"
sys.path.append(COMFY_UI_REPO_PATH)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # for lib

from tqdm import tqdm

from lib import iter_dir_for_video_and_mask, get_video_stats, ensure_video_and_mask_match
from wan_1_3B import run_inpainting_wan_1_3
from wan_14B import run_inpainting_wan_14


def run_model(
    model_name: str,
    model_func,
    input_dir: str,
    output_root: str,
):
    output_dir = os.path.join(output_root, model_name)
    os.makedirs(output_dir, exist_ok=True)
    # run once to ensure video and mask match and the program doesn't crash half way through
    for video in tqdm(iter_dir_for_video_and_mask(input_dir), desc="Checking video and mask"):
        video_stats = get_video_stats(video["video"])
        mask_stats = get_video_stats(video["mask"])
        mask_stats.frame_count = video_stats.frame_count  # TODO: HACK
        ensure_video_and_mask_match(video_stats, mask_stats)
        # assert video_stats.width == output_width
        # assert video_stats.height == output_height

    # do the actual inference
    for video in tqdm(iter_dir_for_video_and_mask(input_dir), desc=f"Running inference for {model_name}"):
        video_stats = get_video_stats(video["video"])

        wan_out_fp = model_func(
            input_video=video["video"],
            input_mask=video["mask"],
            output_prefix=model_name,
            output_frames=30  # TODO: Use frame count
        )
        wan_out_fp = os.path.join(COMFY_UI_REPO_PATH, "output", wan_out_fp)
        out_fp = os.path.join(output_dir, f"{video['video_name']}.mp4")
        os.rename(wan_out_fp, out_fp)


if __name__ == "__main__":
    output_width, output_height = 320, 240
    input_dir = "../temp/inputs"
    output_dir = "../temp/outputs"
    model_runs = [
        # {
        #     "name": "wan_1.3",
        #     "function": run_inpainting_wan_1_3
        # },
        {
            "name": "wan_14",
            "function": run_inpainting_wan_14
        }
    ]

    for model in model_runs:
        print(f"Running model: {model['name']}")
        run_model(
            model_name=model["name"],
            model_func=model["function"],
            input_dir=input_dir,
            output_root=output_dir
        )
