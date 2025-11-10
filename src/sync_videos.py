"""
This script trims the first few seconds off the color video so it matches the
depth video.

Reason:
In my dataset, the depth video had random glitches and delays — it ended up about
6 seconds shorter than the color video overall. Since the depth video glitched
mostly at the start, I decided to trim a few seconds (3s) from the color video
so both videos align better when processed together.
"""

import cv2
import argparse
import os
from tqdm import tqdm


def ensure_parent(path):
    """
    Makes sure the folder exists else it creates it automatically
    """
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def trim_color_video(color_path, out_color_path, trim_seconds=3):
    """
    Trims the first n seconds from a color video and saves the result

    - color_path: input video file
    - out_color_path: where to save the trimmed video
    - trim_seconds: how many seconds to cut from the start (default: 3)
    """

    # Open the input video
    cap = cv2.VideoCapture(color_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open color video: {color_path}")

    # Get video info like FPS and frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    trim_frame = int(round(trim_seconds * fps)) # How many frames to skip
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    remaining_frames = total_frames - trim_frame

    print(f"Trimming first {trim_seconds}s ({trim_frame} frames) from color video")

    # Move the video reader forward by trim_frame frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, trim_frame)

    # Get frame size (width and height)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up the video writer and save as .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ensure_parent(out_color_path)
    out = cv2.VideoWriter(out_color_path, fourcc, fps, (w, h))

    # Loop through all remaining frames and write them to new file
    for _ in tqdm(range(remaining_frames), desc="Writing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Clean up
    cap.release()
    out.release()

    print(f"Trimmed color video saved as {out_color_path} (length ~{remaining_frames/fps:.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--color', required=True, help='path to input color video')
    parser.add_argument('--out_color', required=True, help='path to save trimmed video')
    parser.add_argument('--trim_seconds', type=float, default=3.0, help='seconds to trim from start')
    args = parser.parse_args()

    # Run the trimming function with the given arguments
    trim_color_video(args.color, args.out_color, trim_seconds=args.trim_seconds)
