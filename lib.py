import os
import cv2
from dataclasses import dataclass


@dataclass
class VideoStats:
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int


def get_video_stats(path: str) -> VideoStats:
    """Get video statistics such as width, height, fps, duration, and frame count.

    Args:
        path (str): Path to the video file.

    Returns:
        VideoStats: A dataclass containing video statistics.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return VideoStats(width=width, height=height, fps=fps, duration=duration, frame_count=frame_count)


def ensure_video_and_mask_match(video_stats: VideoStats, mask_stats: VideoStats) -> None:
    """Ensure that the video and mask have matching dimensions and frame counts.

    Args:
        video_stats (VideoStats): The statistics of the video.
        mask_stats (VideoStats): The statistics of the mask.

    Raises:
        ValueError: If the video and mask do not match in dimensions or frame count.
    """
    assert video_stats.width == mask_stats.width, f"Video width {video_stats.width} does not match mask width {mask_stats.width}"
    assert video_stats.height == mask_stats.height, f"Video height {video_stats.height} does not match mask height {mask_stats.height}"
    assert video_stats.frame_count == mask_stats.frame_count, f"Video frame count {video_stats.frame_count} does not match mask frame count {mask_stats.frame_count}"



def iter_dir_for_video_and_mask(
    dir: str,
    video_dir: str = "originals",
    mask_dir: str = "masks",
    video_suffix: str = ".avi",
    mask_suffix: str = "_mask.avi",
) -> list[dict[str, str]]:
    """Iterate through a directory and find video and mask pairs.
    Returns:
        list[dict[str, str]]: A list of dictionaries containing video and mask paths.
    """
    videos = []
    dir = os.path.abspath(dir)  # normalise the base directory
    video_dir = os.path.join(dir, video_dir)
    mask_dir = os.path.join(dir, mask_dir)
    for fname in os.listdir(video_dir):
        assert fname.endswith(video_suffix) and not fname.endswith(mask_suffix)
        video_name = fname.replace(video_suffix, "")
        video_path = os.path.abspath(os.path.join(video_dir, fname))
        mask_path = os.path.abspath(os.path.join(mask_dir, video_name + mask_suffix))
        video = {"video_name": video_name, "video": video_path, "mask": mask_path}
        assert os.path.exists(video["mask"]), f"Mask file does not exist: {video['mask']}"
        videos.append(video)
    return videos


# def iter_dir_for_video_and_mask(
#     dir: str,
#     video_suffix: str = ".avi",
#     mask_suffix: str = "_mask.avi",
# ) -> list[dict[str, str]]:
#     """Iterate through a directory and find video and mask pairs.

#     Args:
#         dir (str): The directory to search.
#         video_suffix (str, optional): The suffix for video files. Defaults to ".avi".
#         mask_suffix (str, optional): The suffix for mask files. Defaults to "_shake_mask.avi".

#     Returns:
#         list[dict[str, str]]: A list of dictionaries containing video and mask paths.
#     """
#     videos = []
#     dir = os.path.abspath(dir)  # normalise the base directory
#     for fname in os.listdir(dir):
#         if fname.endswith(video_suffix) and not fname.endswith(mask_suffix):
#             video_name = fname.replace(video_suffix, "")
#             video_path = os.path.abspath(os.path.join(dir, fname))
#             mask_path = os.path.abspath(os.path.join(dir, video_name + mask_suffix))
#             video = {"video_name": video_name, "video": video_path, "mask": mask_path}
#             assert os.path.exists(video["mask"]), f"Mask file does not exist: {video['mask']}"
#             videos.append(video)
#     return videos
