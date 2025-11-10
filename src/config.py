"""
Configuration and thresholds (metric system — meters, m/s)
----------------------------------------------------------
These values are tuned for Intel RealSense D455 outputs,
where depth is measured in meters (float32).
"""

# --- Detection ---
DETECT_CONF = 0.35 # Minimum 35% confidence for YOLO detections, else just ignore
DETECT_CLASSES = ('person', 'car', 'bicycle', 'motorbike')  # Only track objects we care about

# --- Tracking ---
IOU_MATCH_THRESHOLD = 0.25 # Minimum IoU needed to consider a detection the same object as a previous one
TRACK_HISTORY_LEN = 10 # Number of frames we remember for each object

# --- Depth / Safety thresholds ---
# Distances in meters, velocities in meters/second.
# These define alerts for “danger” or “caution” zones.
DANGER_DIST = 1.5 # Object is dangerously close if <= 1.5 meters
CAUTION_DIST = 3.0 # Object is in caution zone if <= 3 meters

DANGER_VEL = 1.5 # Object moving fast if >= 1.5 m/s
CAUTION_VEL = 0.5 # Object moving slowly but approaching if >= 0.5 m/s

# --- Visualization ---
BUBBLE_RADIUS_PIXELS = 120 # How big the safety bubble looks on screen (pixels)
BUBBLE_CENTER_OFFSET_Y = 0.25 # Vertical position of bubble center

# --- Logging / Output ---
LOG_CSV = "alerts.csv" # File to store alerts with timestamps
OUTPUT_FPS = None # Use same FPS as input video (None) or set manually

# --- Tunable local params ---
BOTTOM_FRACTION = 0.35 # Fraction of bottom part of bounding box to sample for depth
BOTTOM_PERCENTILE = 20 # Pick lower 20th percentile of depth values to estimate closest depth
TTC_THRESHOLD = 3.0 # Time-to-collision threshold (seconds) for alerts
CENTER_STRIP_RATIO = 0.10 # Only objects in the center 10% of frame width are considered in center path
CONSECUTIVE_FRAMES_TO_ALERT = 3 # Number of frames an object must continuously trigger danger to alert
CONSECUTIVE_FRAMES_TO_CLEAR = 3 # Number of frames without danger before clearing the alert