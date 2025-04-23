import os
import cv2
import numpy as np
import glob

############################
# 1) Simple IoU Tracker
############################
class Track:
    """
    A simple structure for a single tracked object.
    Stores current bounding box and track ID.
    """
    def __init__(self, track_id, x1, y1, x2, y2, score, cls):
        self.track_id = track_id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.cls = cls
        self.lost_count = 0  # how many consecutive frames it's gone unmatched

def compute_iou(boxA, boxB):
    """
    boxA, boxB in [x1, y1, x2, y2] format
    returns IoU in [0..1]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = areaA + areaB - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

class IOUTracker:
    """
    A naive IoU-based multi-object tracker:
      - Maintains a list of active tracks
      - For each new frame's detections, tries to match them to existing tracks by maximum IoU
      - If IoU > iou_threshold, treat it as a match
      - If no match, create a new track
      - Unmatched tracks are removed or marked lost. (Here, we remove them immediately)
    """
    def __init__(self, iou_threshold=0.5, max_lost=1):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost  # how many frames we allow a track to go unmatched
        self.tracks = []          # list of Track objects
        self.next_id = 1          # next track ID to assign

    def update(self, detections):
        """
        detections: Nx6 array => [x1, y1, x2, y2, score, cls]
        returns a Nx7 array => [x1, y1, x2, y2, track_id, cls, score]
        """
        # Step 1: Mark all tracks as unmatched
        for t in self.tracks:
            t.lost_count += 1

        # Step 2: For each detection, find the best track by IoU
        used_dets = set()
        used_tracks = set()

        detection_array = []
        for d_idx, det in enumerate(detections):
            x1, y1, x2, y2, score, cls = det
            best_iou = 0.0
            best_track = None
            best_t_idx = None

            for t_idx, track in enumerate(self.tracks):
                if t_idx in used_tracks:
                    continue
                # compute IoU
                iou_val = compute_iou([track.x1, track.y1, track.x2, track.y2],
                                      [x1, y1, x2, y2])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_track = track
                    best_t_idx = t_idx

            if best_track is not None and best_iou >= self.iou_threshold:
                # match found => update track
                best_track.x1 = x1
                best_track.y1 = y1
                best_track.x2 = x2
                best_track.y2 = y2
                best_track.score = score
                best_track.cls = cls
                best_track.lost_count = 0
                # mark them used
                used_dets.add(d_idx)
                used_tracks.add(best_t_idx)

        # Step 3: Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx in used_dets:
                continue
            x1, y1, x2, y2, score, cls = det
            new_track = Track(self.next_id, x1, y1, x2, y2, score, cls)
            self.next_id += 1
            self.tracks.append(new_track)

        # Step 4: Remove tracks that remain unmatched too long
        alive_tracks = []
        for track in self.tracks:
            if track.lost_count <= self.max_lost:
                alive_tracks.append(track)
        self.tracks = alive_tracks

        # Step 5: Build output array with current tracks
        outputs = []
        for t in self.tracks:
            # [x1, y1, x2, y2, track_id, cls, score]
            outputs.append([t.x1, t.y1, t.x2, t.y2, t.track_id, t.cls, t.score])

        return np.array(outputs, dtype=np.float32)


############################
# 2) Load Detections in MOT Format
############################
def load_mot_detections(det_path):
    """
    det.txt typically has lines like:
       frame, -1, x, y, w, h, score, -1, -1
    We'll parse them into a dict: detections_by_frame[frame_id] = list of [x1, y1, x2, y2, score, cls(=0)]
    """
    detections_by_frame = {}
    with open(det_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # e.g. [frame, ?, x, y, w, h, score, ?, ?]
            frame_id = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = float(parts[6])

            if w <= 0 or h <= 0:
                continue
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            cls = 0  # default class label
            det = [x1, y1, x2, y2, score, cls]
            detections_by_frame.setdefault(frame_id, []).append(det)
    return detections_by_frame

############################
# 3) Main Tracking + Visualization
############################
def run_iou_tracker(seq_dir, output_video, fps=30, iou_threshold=0.5):
    """
    seq_dir: e.g. "MOT17-11-SDP" containing:
       - img1 folder with frames
       - det/det.txt
    output_video: e.g. "iou_output.avi"
    fps: frame rate for the output video
    iou_threshold: threshold for matching
    """
    img_dir = os.path.join(seq_dir, "img1")
    det_path = os.path.join(seq_dir, "det", "det.txt")

    # 1) Load all frames
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_files:
        print(f"No images found in {img_dir}")
        return

    # 2) Load detections
    if not os.path.exists(det_path):
        print(f"No det.txt found at {det_path}")
        return
    detections_by_frame = load_mot_detections(det_path)

    # 3) Initialize IoU tracker
    tracker = IOUTracker(iou_threshold=iou_threshold, max_lost=1)

    # 4) Prepare output video
    first_frame = cv2.imread(img_files[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"Running IOU Tracker on {len(img_files)} frames with iou_threshold={iou_threshold}...")
    for idx, img_path in enumerate(img_files, start=1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: couldn't read {img_path}. Skipping.")
            continue

        # Gather detections for frame idx
        frame_dets = detections_by_frame.get(idx, [])
        if frame_dets:
            detection_array = np.array(frame_dets, dtype=np.float32)
        else:
            detection_array = np.zeros((0,6), dtype=np.float32)

        # Update tracker
        tracks_out = tracker.update(detection_array)
        # tracks_out is Nx7 => [x1, y1, x2, y2, track_id, cls, score]

        # Draw results
        for t in tracks_out:
            x1, y1, x2, y2, track_id, cls, score = t
            color = (0,255,0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID:{int(track_id)}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_video.write(frame)

    out_video.release()
    print(f"IOU tracking complete. Output saved to {output_video}")


###################################
# Example usage (no argparse)
###################################
if __name__ == "__main__":
    seq_dir = "Tracking/Tracking/train/MOT17-11-SDP" 
    output_video = "iou_output.mp4"
    fps = 30
    iou_threshold = 0.5

    run_iou_tracker(seq_dir, output_video, fps, iou_threshold)
