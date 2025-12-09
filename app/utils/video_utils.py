# app/utils/video_utils.py

import requests
import subprocess
import uuid
import cv2
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple, Optional


TMP_DIR = Path("tmp_videos")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def save_upload_file(upload_file, destination: Path) -> Path:
    """
    Save an UploadFile to disk.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        for chunk in upload_file.file:
            f.write(chunk)
    return destination


def save_bytes_to_file(data: bytes, destination: Path) -> Path:
    """
    Save raw bytes to file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        f.write(data)
    return destination


def download_video(url: str, destination: Path) -> Path:
    """
    Downloads a video from URL to destination.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=30)
    if r.status_code != 200:
        raise Exception("Failed to download video")

    with destination.open("wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    return destination


def convert_to_mp4(input_path: Path, output_path: Path) -> Path:
    """
    Converts any input video to MP4 using ffmpeg.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-preset", "veryfast",
        "-movflags", "+faststart",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # If ffmpeg fails, return original path
        return input_path

    return output_path


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """
    Returns {width, height, fps, frame_count}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception("Could not open video")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

    cap.release()
    return info


def frame_generator(video_path: Path, start: int = 0, end: Optional[int] = None
                    ) -> Iterator[Tuple[int, Any]]:
    """
    Yields (frame_index, frame_BGR) for each frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception("Could not open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None:
        end = total

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    idx = start
    while idx < end:
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame
        idx += 1

    cap.release()


def make_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """
    Creates a VideoWriter for writing annotated MP4 videos.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise Exception("Could not create output video")

    return writer


def safe_remove(path: Path):
    """Remove a file if exists (ignore errors)."""
    try:
        if path.exists():
            path.unlink()
    except:
        pass
