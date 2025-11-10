"""
A simple YOLOv8 object detector.

It tries to load a YOLO model and use it to detect objects in an image frame.
If YOLO isn’t installed or fails to load, the detector just returns no detections
instead of crashing the program.
"""

from typing import List, Tuple
import logging
import numpy as np

# Handle missing dependencies instead of crashing
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# Set up a logger for debugging/info
log = logging.getLogger("detector")


class Detector:
    """
    Class for the YOLOv8 object detection model
    """

    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.35):
        """
        model_name: str = "yolov8n.pt" - Which YOLO model to load (default = yolov8n.pt)
        conf_thresh: float = 0.35 - Ignore predictions below 35% confidence
        """

        # Start with no model loaded
        self.model = None
        # Save the minimum confidence
        self.conf = conf_thresh

        # Check if YOLO is available before trying to load
        if YOLO_AVAILABLE:
            try:
                # Load YOLO model
                self.model = YOLO(model_name)
                log.info(f"Loaded YOLO model {model_name}")
            except Exception as e:
                # Handle model loading failures
                log.warning(f"Failed to load YOLO model: {e}")
                self.model = None
        else:
            log.warning("Ultralytics YOLO not available. Detector will not run.")

    def detect(self, frame: np.ndarray) -> List[Tuple[List[float], str, float]]:
        """
        Runs object detection on one image frame
        """

        # If the model failed to load, just return nothing
        if self.model is None:
            return []

        # Run YOLO detection on the image frame
        # imgsz=640 - resizes the image internally to 640x640
        # conf=self.conf - the minimal confidence
        results = self.model(frame, imgsz=640, conf=self.conf)[0]

        detections = []

        # Loop through each detected object on the frame
        for box in results.boxes:
            # Box coordinates: [x1, y1, x2, y2]
            xyxy = box.xyxy.cpu().numpy().reshape(4).tolist()

            # Confidence score - how sure the model is about this detection
            conf = float(box.conf.cpu().numpy())

            # Class index (like 0 = person, 1 = car, and so on)
            cls_idx = int(box.cls.cpu().numpy())

            # Get the class name (like person) from the model if its available
            if hasattr(self.model, "names"):
                cls_name = self.model.names[cls_idx]
            else:
                cls_name = str(cls_idx)

            # Add this detection to our list
            detections.append((xyxy, cls_name, conf))

        # Return all detections for this frame
        return detections