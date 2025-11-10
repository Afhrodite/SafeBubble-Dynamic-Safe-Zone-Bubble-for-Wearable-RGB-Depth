"""
Automatically grabs 10 'alert' and 10 'safe' frames
"""

import cv2
import os
import numpy as np

# CONFIG
VIDEO_PATH = "../outputs/demo_out_segment.mp4" # Our processed video
OUT_ALERT = "../outputs/readme_alert_samples" # Where alert frames will go
OUT_SAFE = "../outputs/readme_safe_samples" # Where safe frames will go
SAMPLE_EVERY = 2 # Check every 2nd frame
NUM_SAMPLES = 10 # How many samples to save per category
RED_THRESH = 140 # Pixel needs to be >140 to be considered strong red
RED_DOM_RATIO = 1.15 # R must be 15% higher than G to count as "dominant"
MIN_FRAME_SPACING = 20 # Avoid saving nearly identical frames

# Make sure output folders exist
os.makedirs(OUT_ALERT, exist_ok=True)
os.makedirs(OUT_SAFE, exist_ok=True)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Opened {VIDEO_PATH} with {total_frames} frames")

scores = []  # Store pairs here

# Loop through the video and compute a red intensity score for each frame
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends

    # Only process every Nth frame to save time
    if frame_idx % SAMPLE_EVERY == 0:
        b, g, r = cv2.split(frame)  # Split into color channels

        # Compute how much red is in this frame
        r_mean = float(r.mean())  # Average red intensity
        # Find pixels that are strong red
        strong_mask = (r > RED_THRESH) & (r > (g * RED_DOM_RATIO))
        # Proportion of strong red pixels in the frame
        strong_prop = float(np.count_nonzero(strong_mask)) / (frame.shape[0] * frame.shape[1])

        # Combine both metrics into one score
        score = r_mean * 0.7 + strong_prop * 255.0 * 0.3
        scores.append((score, frame_idx))

    frame_idx += 1

cap.release()
print(f"Computed alert scores for {len(scores)} frames")

# Sort frames by redness (lower = safe, higher = alert)
scores_sorted = sorted(scores, key=lambda x: x[0])
safe_candidates = scores_sorted[:NUM_SAMPLES * 3]     # Bottom (less red)
alert_candidates = scores_sorted[-NUM_SAMPLES * 3:]   # Top (most red)

def pick_spaced(samples, n, min_spacing):
    """
    Helper function to avoid picking frames too close together
    """
    selected = []
    # Sort by score descending (most red first)
    for s in sorted(samples, key=lambda x: x[0], reverse=True):
        # Make sure it's not too close to already chosen frames
        if all(abs(s[1] - sel[1]) > min_spacing for sel in selected):
            selected.append(s)
        if len(selected) >= n:
            break
    return selected

# Pick 10 alert and 10 safe frames nicely spaced apart
alert_final = pick_spaced(alert_candidates, NUM_SAMPLES, MIN_FRAME_SPACING)
safe_final = pick_spaced(safe_candidates, NUM_SAMPLES, MIN_FRAME_SPACING)

print(f"Selected {len(alert_final)} alert and {len(safe_final)} safe frames")

# Save those frames to disk
cap = cv2.VideoCapture(VIDEO_PATH)

# Save alert frames
for rank, (score, idx) in enumerate(alert_final):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    if not ret:
        continue
    out_path = os.path.join(OUT_ALERT, f"alert_{rank+1:02d}_frame{idx}_score{int(score)}.jpg")
    cv2.imwrite(out_path, frame)

# Save safe frames
for rank, (score, idx) in enumerate(safe_final):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    if not ret:
        continue
    out_path = os.path.join(OUT_SAFE, f"safe_{rank+1:02d}_frame{idx}_score{int(score)}.jpg")
    cv2.imwrite(out_path, frame)

cap.release()

print("Done!")
print(f"Saved {len(alert_final)} alert and {len(safe_final)} safe frames.")
