"""
@author: K. Danielewski
"""

import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_dir_save(save_path: str, name: str) -> str:
    """Auxfun. Checks if dir of joined path exists, if not makes dir, else outputs joined path"""
    if not os.path.exists(os.path.join(save_path, name)):
        os.mkdir(os.path.join(save_path, name))
        save_here = os.path.join(save_path, name)
    else:
        save_here = os.path.join(save_path, name)
    return save_here


def extract_first_frames(video_paths: list[str]) -> None:
    """Extracts the first frame from each .mp4 file and saves it as a .png next to the video.

    Args:
        video_paths: list of paths to .mp4 files
    """
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Could not read frame from {path}")
            continue
        out_path = Path(path).with_suffix(".png")
        cv2.imwrite(str(out_path), frame)


def measure_line_to_height(png_paths: list[str]) -> dict[str, float]:
    """Opens each PNG, lets the user draw a line by clicking two points, and returns
    the ratio of line length to image height for each file.

    Click once to set the start point and once more to set the end point.
    The line is shown on the image before moving to the next file.

    Args:
        png_paths: list of paths to PNG files

    Returns:
        dict mapping filename stem to (line_length / image_height)
    """
    results = {}

    for path in png_paths:
        img_bgr = cv2.imread(str(path))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height = img.shape[0]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)
        ax.set_title(f"{Path(path).name} — click two points to draw a line, then close")
        ax.axis("off")
        plt.tight_layout()

        points = plt.ginput(2, timeout=0)

        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            ratio = line_length / height

            ax.plot([x1, x2], [y1, y2], color="red", linewidth=2)
            ax.scatter([x1, x2], [y1, y2], color="red", s=40, zorder=5)
            ax.set_title(
                f"{Path(path).name} — ratio: {ratio:.4f}  (close to continue)"
            )
            plt.draw()
            plt.waitforbuttonpress()

            results[Path(path).stem] = ratio
        else:
            print(f"No line drawn for {Path(path).name}, skipping.")

        plt.close(fig)

    return results


def check_data_folder(data_folder: str | list) -> str:
    """Auxfun to check whether data folder was passed correctly. If str, glob to take all dirs inside"""
    try:
        if isinstance(data_folder, list):
            pass
        elif isinstance(data_folder, str):
            data_folder = glob(data_folder + "\\*")
    except:
        print(f"Passed data folder should be either string or a list, {type(data_folder)} was passed")
    return data_folder
