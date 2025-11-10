"""
Helper functions to work with depth images

These functions help measure things like:
    - the median depth inside a bounding box
    - calibrating pixel depth values to real distances
    - and finding the closest depth near the bottom of an object box
"""

import numpy as np


def median_depth_in_bbox(depth_frame: np.ndarray, bbox, invalid_val=0):
    """
    Finds the median depth value inside a bounding box area of the depth frame

    - depth_frame: single-channel image (so we would get 1 value per pixel)
    - bbox: the bounding box [x1, y1, x2, y2]
    - invalid_val=0: no data when the pixel value is 0

    returns: the median depth value or None
    """

    # Make sure the coordinates are integers
    x1,y1,x2,y2 = map(int, bbox)

    # Get frame height and width
    h,w = depth_frame.shape[:2]

    # Check coordinates so they dont go outside the image
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    # If the box is invalid - stop
    if x2<=x1 or y2<=y1:
        return None

    # Crop the region of interest (ROI) from the depth image
    roi = depth_frame[y1:y2, x1:x2]

    # Flatten to a 1D array so it's easier to process
    flat = roi.flatten()

    # Remove invalid depth values (like zeros)
    if invalid_val is not None:
        flat = flat[flat != invalid_val]

    # If there are no valid pixels, return None
    if flat.size == 0:
        return None

    # Return the middle value of all valid depths
    return float(np.median(flat))

def basic_linear_calibrate(sample_pairs):
    """
    Does a simple linear calibration between depth pixels and real distances

    - sample_pairs: list of (pixel_value, real_distance_m)
        - pixel_value: what the depth sensor gave
        - real_distance_m: measured true distance in meters(m)
    """

    # Split the sample pairs into X (pixels) and Y (distances)
    X = np.array([p for p, _ in sample_pairs], dtype=np.float32)
    Y = np.array([d for _, d in sample_pairs], dtype=np.float32)

    # Create the linear model matrix for y = a*x + b
    A = np.vstack([X, np.ones_like(X)]).T

    # Find the best line parameters
    sol, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    scale, offset = sol[0], sol[1]

    # Return the line parameters so we can convert pixel to real distance
    return scale, offset

def closest_depth_in_bottom_region(depth_frame: np.ndarray,
                                   bbox,
                                   bottom_fraction=0.35,
                                   percentile=20,
                                   invalid_vals=(0,),
                                   pad_px=8):
    """
    Tries to estimate the smallest distance inside the lower part of the bounding box
    Estimating how far an object is from the camera — especially the part touching the ground

    - bottom_fraction: 35% of the box from the bottom to look at
    - percentile: which percentile to take (smaller = closer distance)
    - invalid_vals: pixel values that doesnt have any depth info
    - pad_px: expand the box a bit to include nearby pixels

    returns: a float depth value or None if it couldn’t find a valid one
    """

    # Convert coordinates to integers
    x1, y1, x2, y2 = map(int, bbox)
    h_img, w_img = depth_frame.shape[:2]

    # Pad the box - helps if the box edges are off by a few pixels
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(w_img - 1, x2 + pad_px)
    y2 = min(h_img - 1, y2 + pad_px)

    # Make sure the box is valid
    if x2 <= x1 or y2 <= y1:
        return None

    # Calculate how tall the bottom region should be
    bbox_h = y2 - y1
    bottom_h = max(1, int(round(bbox_h * bottom_fraction)))

    # Where the bottom region starts
    by1 = max(y1, y2 - bottom_h)

    # Crop the bottom region from the depth frame
    roi = depth_frame[by1:y2, x1:x2]

    # Flatten it for easier processing
    flat = roi.flatten().astype(float)

    # If no pixels are found, try a smaller column in center
    if flat.size == 0:
        cx = int((x1 + x2) / 2)
        col_w = max(2, int((x2-x1)*0.1))
        cx1 = max(0, cx - col_w); cx2 = min(w_img, cx + col_w)
        roi2 = depth_frame[by1:y2, cx1:cx2].flatten().astype(float)
        flat = roi2

    # If its still empty, just return None
    if flat.size == 0:
        return None

    # Remove invalid values
    mask = np.ones_like(flat, dtype=bool)
    for iv in invalid_vals:
        mask &= (flat != iv)
    flat_valid = flat[mask]

    # If no valid values left, fall back to using everything
    if flat_valid.size == 0 and flat.size > 0:
        flat_valid = flat  # just to return something

    # If still nothing, just return None
    if flat_valid.size == 0:
        return None

    # Take the lower percentile — smaller = closer distance
    try:
        val = float(np.percentile(flat_valid, percentile))
    except:
        val = float(np.median(flat_valid)) # If percentile fails, just use the median instead
    return val