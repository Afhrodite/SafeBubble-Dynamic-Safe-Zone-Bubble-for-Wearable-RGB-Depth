"""
Helpers to draw annotated frames and safety bubble for visualization.
"""
import cv2
import numpy as np


def depth_to_vis(depth_frame):
    """
    Convert a single-channel depth frame to a nice color map for visualization.

    - depth_frame: 2D array of depth values in meters

    Returns: 3-channel BGR image
    """

    # Normalize depth values to 0–255 range
    vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # Apply a colormap so closer/farther objects are visually distinct
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    return vis

def draw_bubble_and_dets(color_frame, depth_vis, detections_info, bubble_center, bubble_radius, bubble_color):
    """
    Overlay the safety bubble and detected objects on a color frame.
    Also show depth visualization side-by-side.

    - detections_info: list of (bbox, class_name, id, dist, vel, alert_level)
    - bubble_center: (x,y) - the center in pixels
    - bubble_radius: radius of safety bubble in pixels
    - bubble_color: BGR tuple

    Returns combined image (side-by-side color + depth visualization)
    """
    # Work on a copy so original frame is not changed
    out = color_frame.copy()

    # Draw the safety bubble
    cv2.circle(out, bubble_center, int(bubble_radius), bubble_color, thickness=2)

    # Draw bounding boxes and labels for each detection
    for bbox, cls_name, tid, dist, vel, alert in detections_info:
        x1,y1,x2,y2 = map(int, bbox)

        # Choose color based on alert level
        if alert == 'danger':
            col = (0,0,255) # red
        elif alert == 'caution':
            col = (0,165,255) # orange
        else:
            col = (0,255,0) # green

        # Draw the bounding box
        cv2.rectangle(out, (x1,y1),(x2,y2), col, 2)

        # Build label text
        label = f"{cls_name} id{tid}"
        if dist is not None:
            label += f" d={dist:.1f}" # distance in m
        if vel is not None:
            label += f" v={vel:+.1f}" # velocity in m/s

        # Draw text above the bounding box
        cv2.putText(out, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # Combine color + depth visualization side-by-side
    combined = np.hstack((out, depth_vis))
    return combined