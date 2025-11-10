"""
Orchestrates the SafeBubble pipeline.

This script puts everything together:
    - read color + depth videos
    - run detector (YOLOv8) on color frames
    - track objects across frames
    - estimate depth/velocity/ttc and decide alerts
    - draw a safety bubble + annotated output
    - save annotated video and CSV logs

Example usage:
--------------
python main.py \
    --color ../original_data/synced_color.mp4 \
    --depth ../original_data/challenge_depth_848x480.mp4 \
    --out ../outputs/demo_out_segment.mp4 \
    --view \
    --start-time 00:12 \
    --end-time 02:34 \
    --smooth-alpha 0.4
"""
import argparse
import os
import time
import cv2
import numpy as np
import pandas as pd

from config import *
from detector import Detector
from tracker import SimpleTracker
from depth_utils import median_depth_in_bbox, closest_depth_in_bottom_region
from visualize import depth_to_vis, draw_bubble_and_dets


def parse_time_string(t):
    """
    Convert time input like "mm:ss" or "ss" (or a numeric) to float seconds.
    Returns None if parsing fails or input is None.
    """

    if t is None:
        return None
    if isinstance(t, (float, int)):
        return float(t)
    t = str(t)
    if ':' in t:
        parts = t.split(':')
        # mm:ss
        if len(parts) == 2:
            mm, ss = parts
            return float(mm) * 60.0 + float(ss)
        # hh:mm:ss
        elif len(parts) == 3:
            hh, mm, ss = parts
            return float(hh) * 3600.0 + float(mm) * 60.0 + float(ss)
    try:
        return float(t)
    except:
        return None


def parse_args():
    """
    CLI args for quick testing / demo runs.
    I often pass small segments when testing to save time.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--color', required=True, help='path to color video')
    p.add_argument('--depth', required=True, help='path to depth video')
    p.add_argument('--out', default='../outputs/demo_out.mp4', help='output annotated video')
    p.add_argument('--log', default=LOG_CSV, help='CSV log file')
    p.add_argument('--view', action='store_true', help='show preview window')
    p.add_argument('--max-frames', type=int, default=None, help='process at most N frames (for quick tests)')
    p.add_argument('--warmup-secs', type=float, default=0.0, help='skip first N seconds of processing')
    p.add_argument('--start-time', type=str, default=None, help='start time for demo segment')
    p.add_argument('--end-time', type=str, default=None, help='end time for demo segment')
    p.add_argument('--smooth-alpha', type=float, default=0.6, help='EMA smoothing alpha for depth (0..1)')
    return p.parse_args()

def normalize_depth_frame(depth_frame):
    """
    Quick helper to understand what range the depth frame uses.
    Returns a float32 copy and a small string guess about its unit/range.
    Not critical for pipeline, just handy during debugging.
    """
    df = depth_frame.astype(np.float32)
    mn, mx = float(df.min()), float(df.max())
    if mx > 1000:  # Likely mm or uint16 raw values
        if mx > 2000 and mx < 30000:  # mm values up to 30k
            return df / 1000.0, 'meters_mm'  # Convert mm -> meters
        return df, 'large_values'
    if mx <= 255:
        return df, 'relative_0_255'
    return df, 'unknown'

def ensure_parent_exists(path):
    """
    Make sure output directory exists before writing files
    """
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def decide_alert_simple(dist, vel):
    """
    Simple fallback alert logic.
    Kept for backward compatibility — the main code uses a more detailed decision.
    """

    # Simple rules: close + fast -> danger, close-slower -> caution, otherwise safe
    if dist is None:
        return 'unknown'
    if vel is None:
        vel = 0.0
    if dist <= DANGER_DIST and vel >= DANGER_VEL:
        return 'danger'
    elif dist <= CAUTION_DIST and vel >= CAUTION_VEL:
        return 'caution'
    else:
        return 'safe'


def main():
    # parse CLI and prepare output folder
    args = parse_args()
    ensure_parent_exists(args.out)

    # Open the color and depth videos
    cap_c = cv2.VideoCapture(args.color)
    cap_d = cv2.VideoCapture(args.depth)
    if not cap_c.isOpened():
        print(f"[ERROR] Cannot open color video: {args.color}")
        return
    if not cap_d.isOpened():
        print(f"[ERROR] Cannot open depth video: {args.depth}")
        return

    # Read video metadata
    fps = cap_c.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_color = int(cap_c.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_depth = int(cap_d.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap_c.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = OUTPUT_FPS or fps

    # Figure out frame indices for optional start/end times
    start_seconds = parse_time_string(args.start_time)
    end_seconds = parse_time_string(args.end_time)
    start_frame = int(round(start_seconds * fps)) if start_seconds is not None else 0
    # Default end_frame uses both video lengths (min) to avoid reading past the end
    end_frame = int(round(end_seconds * fps)) if end_seconds is not None else min(total_frames_color, total_frames_depth) - 1

    # Clamp windows so we don't accidentally go out of bounds
    start_frame = max(0, min(start_frame, total_frames_color - 1))
    end_frame = max(0, min(end_frame, total_frames_color - 1))
    if end_frame <= start_frame:
        print(f"[ERROR] Invalid time window: start_frame={start_frame}, end_frame={end_frame}")
        return

    print(f"Processing frames {start_frame} .. {end_frame} (fps={fps}, total_color={total_frames_color}, total_depth={total_frames_depth})")

    # Prepare side-by-side output writer (color | depth_vis)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(args.out, fourcc, out_fps, (w*2, h))
    if not out_writer.isOpened():
        print(f"[ERROR] VideoWriter failed to open for {args.out}")
        return

    # Seek both videos to the start_frame
    cap_c.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_d.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    # Initialize detector and tracker, where Detector is YOLOv8 wrapper, SimpleTracker is IoU tracker
    detector = Detector(model_name='yolov8n.pt', conf_thresh=DETECT_CONF) if 'Detector' in globals() else Detector(conf_thresh=DETECT_CONF)
    tracker = SimpleTracker(iou_thresh=IOU_MATCH_THRESHOLD, history_len=TRACK_HISTORY_LEN)

    # State variables for logging, fps estimation, fallback depth
    logs = [] # to store the CSV logs
    prev_time = time.time()
    fps_smooth = None
    last_depth = None

    # Skip first N seconds of frames
    warmup_frames = int(round(args.warmup_secs * fps)) if args.warmup_secs and args.warmup_secs > 0 else 0
    if warmup_frames > 0:
        print(f"[INFO] Skipping first {warmup_frames} frames of the segment")

    # The main processing loop
    while frame_idx <= end_frame:
        ret_c, frame_c = cap_c.read()
        ret_d, frame_d = cap_d.read()

        # If color frame missing, stop
        if not ret_c:
            print(f"[WARN] Missing color frame at index {frame_idx}. Stopping.")
            break

        # Skip warmup frames
        if frame_idx - start_frame < warmup_frames:
            frame_idx += 1
            continue

        # Depth frame handling
        # - if depth frame is missing, reuse last depth if available or use a neutral gray
        if not ret_d:
            if last_depth is None:
                # If we never had a depth frame
                depth_gray = (127 * np.ones((h, w), dtype=np.uint8))
            else:
                depth_gray = last_depth.copy()
        else:
            # Depth video may be single-channel or already colored; convert to gray if needed
            if frame_d.ndim == 3:
                depth_gray = cv2.cvtColor(frame_d, cv2.COLOR_BGR2GRAY)
            else:
                depth_gray = frame_d.copy()
            last_depth = depth_gray

        # Make a nice visualization for the depth (with color map)
        try:
            depth_vis = depth_to_vis(depth_gray)
        except Exception:
            # Fallback: normalize and apply jet colormap
            dv = cv2.normalize(depth_gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            depth_vis = cv2.applyColorMap(dv, cv2.COLORMAP_JET)

        # Detection
        dets = detector.detect(frame_c) if hasattr(detector, 'detect') else []
        # Keep only classes we configured (like people, cars)
        dets = [d for d in dets if d[1] in DETECT_CLASSES]

        # Tracking
        tracked = tracker.match_and_update(dets, frame_idx)  # returns [(id,bbox,class,score),...]

        # Collect info for visualization + logging per detection
        detections_info = []
        for tid, bbox, cls_name, score in tracked:
            # Depth extraction for this bbox
            # Bottom-region "closest" estimate, it's usually more accurate
            dist_closest = closest_depth_in_bottom_region(
                depth_gray, bbox,
                bottom_fraction=0.35, # Sample bottom 35% of bbox
                percentile=20, # Take lower 20th percentile for robustness
                invalid_vals=(0, 65535), # Common invalid depth markers
                pad_px=10 # Pad to reduce small alignment issues
            )
            if dist_closest is None:
                # If bottom region gave no valid samples, fallback to median of whole bbox
                dist_closest = median_depth_in_bbox(depth_gray, bbox, invalid_val=0)

            # Add raw depth reading to track history
            tracker.add_depth_to_track(tid, dist_closest)

            # EMA smoothing for depth to reduce jitter
            if 'ema_history' not in tracker.tracks[tid]:
                tracker.tracks[tid]['ema_history'] = []
            ema_hist = tracker.tracks[tid]['ema_history']
            prev_ema = ema_hist[-1] if len(ema_hist) > 0 else None
            if dist_closest is not None:
                if prev_ema is None:
                    # Initialize EMA with first valid measurement
                    ema_val = dist_closest
                else:
                    alpha = float(args.smooth_alpha)
                    ema_val = alpha * prev_ema + (1.0 - alpha) * dist_closest
                ema_hist.append(ema_val)
            else:
                # No new measurement, keep previous EMA
                ema_val = prev_ema

            # Keep EMA history bounded
            if len(ema_hist) > TRACK_HISTORY_LEN:
                ema_hist = ema_hist[-TRACK_HISTORY_LEN:]
            tracker.tracks[tid]['ema_history'] = ema_hist

            # Compute radial velocity (positive = approaching)
            vel = None
            if len(ema_hist) >= 2 and fps > 0:
                # Difference over one frame * fps -> meters/second
                vel = (ema_hist[-2] - ema_hist[-1]) * fps

            # Lateral overlap: how much bbox overlaps center path area
            x1, y1, x2, y2 = map(int, bbox)
            bbox_cx = (x1 + x2) / 2.0
            bbox_w = max(1, x2 - x1)
            frame_cx = w / 2.0
            center_strip_half = CENTER_STRIP_RATIO * w
            center_overlap = max(0, min(x2, frame_cx + center_strip_half) - max(x1, frame_cx - center_strip_half))
            overlap_ratio = center_overlap / bbox_w

            # time-to-collision (TTC) = distance / radial_velocity (if approaching)
            ttc = None
            if vel is not None and vel > 0.01 and ema_val is not None and ema_val > 0.001:
                ttc = ema_val / vel

            # Decide alert level using TTC, distance and lateral overlap
            new_alert = 'safe'
            if (ttc is not None and ttc <= 1.0) or (ema_val is not None and ema_val <= DANGER_DIST and overlap_ratio > 0.2):
                new_alert = 'danger'
            elif (ttc is not None and ttc <= TTC_THRESHOLD) or (ema_val is not None and ema_val <= CAUTION_DIST and overlap_ratio > 0.05):
                new_alert = 'caution'
            else:
                new_alert = 'safe'

            # Hysteresis: avoid flickering alerts by requiring consecutive frames
            if 'alert_state' not in tracker.tracks[tid]:
                tracker.tracks[tid]['alert_state'] = 'safe'
                tracker.tracks[tid]['alert_counter'] = 0
                tracker.tracks[tid]['clear_counter'] = 0

            current_state = tracker.tracks[tid]['alert_state']
            if new_alert == current_state:
                tracker.tracks[tid]['alert_counter'] = 0
                tracker.tracks[tid]['clear_counter'] = 0
            else:
                if new_alert in ('danger', 'caution'):
                    tracker.tracks[tid]['alert_counter'] = tracker.tracks[tid].get('alert_counter', 0) + 1
                    tracker.tracks[tid]['clear_counter'] = 0
                    if tracker.tracks[tid]['alert_counter'] >= CONSECUTIVE_FRAMES_TO_ALERT:
                        tracker.tracks[tid]['alert_state'] = new_alert
                        tracker.tracks[tid]['alert_counter'] = 0
                else:
                    tracker.tracks[tid]['clear_counter'] = tracker.tracks[tid].get('clear_counter', 0) + 1
                    tracker.tracks[tid]['alert_counter'] = 0
                    if tracker.tracks[tid]['clear_counter'] >= CONSECUTIVE_FRAMES_TO_CLEAR:
                        tracker.tracks[tid]['alert_state'] = 'safe'
                        tracker.tracks[tid]['clear_counter'] = 0

            final_alert = tracker.tracks[tid]['alert_state']

            # Collect info for visualization and logging
            detections_info.append((bbox, cls_name, tid, ema_val, vel, final_alert))

            logs.append({
                't': frame_idx / fps,
                'frame': frame_idx,
                'id': tid,
                'class': cls_name,
                'dist': float(ema_val) if ema_val is not None else (float(dist_closest) if dist_closest is not None else np.nan),
                'vel': float(vel) if vel is not None else np.nan,
                'alert': final_alert
            })

        # Decide bubble color: red if any danger, orange if any caution, else green
        bubble_center = (w // 2, int(h * (1.0 - BUBBLE_CENTER_OFFSET_Y)))
        bubble_radius = BUBBLE_RADIUS_PIXELS
        bubble_color = (0, 255, 0)
        for info in detections_info:
            if info[5] == 'danger':
                bubble_color = (0, 0, 255)
                break
            elif info[5] == 'caution':
                bubble_color = (0, 165, 255)

        # Visualization: draw bubble + detections and compose side-by-side with depth
        try:
            combined = draw_bubble_and_dets(frame_c, depth_vis, detections_info, bubble_center, bubble_radius, bubble_color)
        except Exception as e:
            print(f"[WARN] draw_bubble_and_dets failed: {e}. Falling back to side-by-side.")
            try:
                if depth_vis.ndim == 2:
                    depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
                else:
                    depth_vis_bgr = depth_vis
                combined = np.hstack((frame_c, depth_vis_bgr))
            except Exception as e2:
                print(f"[ERROR] fallback failed: {e2}")
                combined = None

        # Ensure combined frame has expected size (w*2, h) before writing
        if combined is not None:
            try:
                target_w = int(w * 2); target_h = int(h)
                if combined.shape[1] != target_w or combined.shape[0] != target_h:
                    combined = cv2.resize(combined, (target_w, target_h), interpolation=cv2.INTER_AREA)
            except Exception as re:
                print(f"[WARN] resize failed: {re}; skipping frame {frame_idx}")
                combined = None

        # Overlay fps / alert counts, write output, and optionally preview
        if combined is not None:
            now = time.time()
            inst_fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            fps_smooth = inst_fps if fps_smooth is None else (0.85 * fps_smooth + 0.15 * inst_fps)
            num_danger = sum(1 for x in detections_info if x[5] == 'danger')
            num_caution = sum(1 for x in detections_info if x[5] == 'caution')
            overlay_text = f"FPS: {fps_smooth:.1f}  Alerts: D={num_danger} C={num_caution}"
            cv2.putText(combined, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            try:
                out_writer.write(combined)
            except Exception as write_err:
                print(f"[WARN] VideoWriter failed to write frame {frame_idx}: {write_err}")

            if args.view:
                try:
                    cv2.imshow('SafeBubble', combined)
                    # Press ESC to exit preview early
                    if cv2.waitKey(1) & 0xFF == 27:
                        print("[INFO] ESC pressed - exiting preview.")
                        break
                except Exception:
                    pass
        else:
            print(f"[WARN] Skipping frame {frame_idx} due to visualization issue.")

        # Advance frame index and optionally stop if max_frames reached
        frame_idx += 1
        if args.max_frames is not None and (frame_idx - start_frame) >= args.max_frames:
            print(f"[INFO] Reached max-frames {args.max_frames}; stopping early.")
            break

    # Write logs to CSV (best-effort)
    try:
        df = pd.DataFrame(logs)
        df.to_csv(args.log, index=False)
        print(f"[INFO] Saved log to {args.log}")
    except Exception as e:
        print(f"[WARN] Could not save log: {e}")

    # Clean up
    out_writer.release()
    cap_c.release()
    cap_d.release()
    cv2.destroyAllWindows()
    print("[INFO] Processing complete.")


if __name__ == '__main__':
    main()

