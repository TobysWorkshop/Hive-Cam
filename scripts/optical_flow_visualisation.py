import numpy as np
import cv2
import json
import os

# --- EMA helper ---
class EMA:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.vx = None
        self.vy = None

    def update(self, new):
        if new is None:
            return (self.vx, self.vy)
        if self.vx is None:
            self.vx, self.vy = new
        else:
            self.vx = self.alpha * new[0] + (1 - self.alpha) * self.vx
            self.vy = self.alpha * new[1] + (1 - self.alpha) * self.vy
        return (self.vx, self.vy)

# --- Flow helpers ---
def forward_backward_mask(flow_fwd, flow_bwd, max_err=1.5):
    h, w = flow_fwd.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    xs_f = xs + flow_fwd[..., 0]
    ys_f = ys + flow_fwd[..., 1]

    map_x = xs_f.astype(np.float32)
    map_y = ys_f.astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0].astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR)
    bwd_y = cv2.remap(flow_bwd[..., 1].astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR)

    rx = flow_fwd[..., 0] + bwd_x
    ry = flow_fwd[..., 1] + bwd_y
    err = np.sqrt(rx**2 + ry**2)
    return err <= max_err

def median_flow_in_bbox(flow, valid_mask, bbox, shrink=0.2, min_valid=10):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    sx = int(x1 + w * shrink/2)
    sy = int(y1 + h * shrink/2)
    sw = int(w * (1 - shrink))
    sh = int(h * (1 - shrink))
    if sw <= 0 or sh <= 0:
        return None

    xs = max(0, sx)
    ys = max(0, sy)
    xe = min(flow.shape[1], sx + sw)
    ye = min(flow.shape[0], sy + sh)

    sub_mask = valid_mask[ys:ye, xs:xe]
    if sub_mask.sum() < min_valid:
        return None

    vals = flow[ys:ye, xs:xe][sub_mask]
    vx = np.median(vals[:, 0])
    vy = np.median(vals[:, 1])
    return float(vx), float(vy)

# --- Main visualization ---
def visualize_with_detections(video_path, flow_folder_fwd, flow_folder_bwd, yolo_json_path):
    # Load YOLO JSON
    with open(yolo_json_path, "r") as f:
        all_detections = json.load(f)  # list of frames, each frame is list of detections

    # EMA per detection (track by detection index)
    ema_dict = {}

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Load optical flow for this frame
        flow_fwd_path = os.path.join(flow_folder_fwd, f"flow_fwd_{frame_idx:06d}.npy")
        flow_bwd_path = os.path.join(flow_folder_bwd, f"flow_bwd_{frame_idx:06d}.npy")
        if not os.path.exists(flow_fwd_path) or not os.path.exists(flow_bwd_path):
            print(f"Flow files missing for frame {frame_idx}, skipping")
            frame_idx += 1
            continue

        flow_fwd = np.load(flow_fwd_path)
        flow_bwd = np.load(flow_bwd_path)
        valid_mask = forward_backward_mask(flow_fwd, flow_bwd)

        detections = all_detections[frame_idx]  # list of detections for this frame

        for det_idx, det in enumerate(detections):
            x1, y1, x2, y2 = det[:4]
            bbox = (x1, y1, x2, y2)

            # initialize EMA per detection index
            if (frame_idx, det_idx) not in ema_dict:
                ema_dict[(frame_idx, det_idx)] = EMA(alpha=0.4)

            med_flow = median_flow_in_bbox(flow_fwd, valid_mask, bbox)
            smoothed_flow = ema_dict[(frame_idx, det_idx)].update(med_flow)

            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw flow vector (arrow)
            if smoothed_flow[0] is not None:
                scale = 5.0  # <-- increase this to exaggerate arrows
                center = (int((x1 + x2)/2), int((y1 + y2)/2))
                end = (int(center[0] + smoothed_flow[0]*scale), int(center[1] + smoothed_flow[1]*scale))
                cv2.arrowedLine(frame, center, end, (0, 0, 255), 2, tipLength=0.3)

        cv2.imshow("Frame with Detections and Flow", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "data/videos/videoB.mp4"
    flow_folder_fwd = "data/raft_flows"
    flow_folder_bwd = "data/raft_flows"  # optional, can be None
    yolo_json_path = "data/yolo_detections/bboxes_origLARGE.json"
    visualize_with_detections(video_path, flow_folder_fwd, flow_folder_bwd, yolo_json_path)
