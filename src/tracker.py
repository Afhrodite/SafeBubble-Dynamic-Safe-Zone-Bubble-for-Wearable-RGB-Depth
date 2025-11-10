"""
Simple IoU-based tracker with depth history support.

Keeps track of objects already detected in video frames.
It assigns a unique ID to each object, updates the position and stores recent
depth values. To flag out the same objects - if they overlap enough with the
existing track.
"""

from collections import deque
from typing import List, Tuple


def iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    boxA, boxB: [x1, y1, x2, y2]

    returns a number between 0 and 1
        - 0 for no overlap
        - 1 for perfect overlap
    """

    # Coordinates of intersection box
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Width and height of intersection
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    # Area of each box
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])

    # IoU formula
    denom = float(boxAArea + boxBArea - interArea)
    if denom == 0:
        return 0.0
    return interArea / denom

class SimpleTracker:
    """
    Tracks objects over time using IoU.
    Each object gets a unique ID and remembers recent depth values.
    """
    def __init__(self, iou_thresh=0.25, history_len=10):
        """
        - iou_thresh: minimum overlap to consider same object
        - history_len: number of depth values to remember per object
        """
        self.iou_thresh = iou_thresh
        self.history_len = history_len
        self.next_id = 0 # counter for new object IDs
        self.tracks = {} # stores all tracked objects

    def match_and_update(self, detections: List[Tuple[List[float], str, float]], frame_idx: int):
        """
        Match new detections with existing tracks

        - detections: list of (bbox, class_name, score) for the current frame
        - frame_idx: current frame index

        returns: list of tracked objects with IDs: (id, bbox, class_name, score)
        """
        assigned = {} # Tracks matched with detections
        unmatched_dets = set(range(len(detections))) # Detections not matched yet
        unmatched_tracks = set(self.tracks.keys()) # Old tracks not matched yet

        # Calculate IoU between all detections and existing tracks
        pairs = []
        for di, (bbox, cls, sc) in enumerate(detections):
            for tid in list(self.tracks.keys()):
                pairs.append((iou(bbox, self.tracks[tid]['bbox']), di, tid))

        # Sort pairs by IoU descending with the highest overlap first
        pairs.sort(reverse=True)

        # Assign detections to tracks if IoU > threshold
        for score, di, tid in pairs:
            if score < self.iou_thresh:
                continue
            if di in unmatched_dets and tid in unmatched_tracks:
                assigned[tid] = di
                unmatched_dets.remove(di)
                unmatched_tracks.remove(tid)

        # Update assigned tracks with new bbox/class info
        output = []
        for tid, di in assigned.items():
            bbox, cls, sc = detections[di]
            self.tracks[tid]['bbox'] = bbox
            self.tracks[tid]['class'] = cls
            self.tracks[tid]['last_seen'] = frame_idx
            output.append((tid, bbox, cls, sc))

        # Create new tracks for unmatched detections
        for di in unmatched_dets:
            bbox, cls, sc = detections[di]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                'bbox': bbox,
                'class': cls,
                'history': deque(maxlen=self.history_len),
                'last_seen': frame_idx
            }
            output.append((tid, bbox, cls, sc))

        # Clean-up - Remove tracks not seen for a long time
        to_delete = []
        for tid, t in self.tracks.items():
            if frame_idx - t['last_seen'] > 150:  # Arbitrary timeout
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return output

    def add_depth_to_track(self, tid: int, depth_value: float):
        """
        Add a new depth reading to a track.
        Only store non-None values.
        """
        if tid in self.tracks:
            if depth_value is not None:
                self.tracks[tid]['history'].append(depth_value)

    def get_last_depth(self, tid: int):
        """
        Get the most recent depth value for a track.
        Returns None if no depth available.
        """
        if tid in self.tracks and len(self.tracks[tid]['history']) > 0:
            return float(self.tracks[tid]['history'][-1])
        return None

    def get_prev_depth(self, tid: int):
        """
        Get the previous depth value for a track (second-to-last reading).
        Useful to check how depth changed between frames.
        """
        if tid in self.tracks and len(self.tracks[tid]['history']) > 1:
            return float(self.tracks[tid]['history'][-2])
        return None