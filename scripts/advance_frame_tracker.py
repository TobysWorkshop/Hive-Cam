import numpy as np
import cv2
import json
import os
import random
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
# --- EMA helper for smoothing optical flow ---
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
# --- Optical flow helpers ---
def forward_backward_mask(flow_fwd, flow_bwd, max_err=1.5):
    h, w = flow_fwd.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    xs_f = xs + flow_fwd[...,0]
    ys_f = ys + flow_fwd[...,1]
    map_x = xs_f.astype(np.float32)
    map_y = ys_f.astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[...,0].astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
    bwd_y = cv2.remap(flow_bwd[...,1].astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR)
    rx = flow_fwd[...,0] + bwd_x
    ry = flow_fwd[...,1] + bwd_y
    err = np.sqrt(rx**2 + ry**2)
    return err <= max_err
def median_flow_in_bbox(flow, valid_mask, bbox, shrink=0.2, min_valid=10):
    x1, y1, x2, y2 = bbox
    w, h = x2-x1, y2-y1
    sx = int(x1 + w*shrink/2)
    sy = int(y1 + h*shrink/2)
    sw = int(w*(1-shrink))
    sh = int(h*(1-shrink))
    if sw <=0 or sh <=0:
        return None
    xs = max(0,sx)
    ys = max(0,sy)
    xe = min(flow.shape[1], sx+sw)
    ye = min(flow.shape[0], sy+sh)
    sub_mask = valid_mask[ys:ye, xs:xe]
    if sub_mask.sum() < min_valid:
        return None
    vals = flow[ys:ye, xs:xe][sub_mask]
    vx = np.median(vals[:,0])
    vy = np.median(vals[:,1])
    return float(vx), float(vy)
# --- Track class ---
class Track:
    def __init__(self, tid, color):
        self.tid = tid
        self.points = []
        self.flow = EMA(alpha=0.4)
        self.pos_vel = EMA(alpha=0.4)
        self.color = color
        self.active = True
        self.pending_deactivation = False
# --- Cost function ---
def compute_directional_cost(track, detection, dt=1.0):
    if not track.points:
        return np.inf
    last_pos = np.array(track.points[-1])
    if len(track.points) > 1 and track.pos_vel.vx is not None:
        vx, vy = track.pos_vel.vx, track.pos_vel.vy
    else:
        vx, vy = track.flow.vx, track.flow.vy
    if vx is None:
        expected = last_pos
    else:
        expected = last_pos + dt * np.array([vx, vy])
    dist = np.linalg.norm(detection - expected)
    angle_penalty = 0
    if vx is not None:
        motion_vec = np.array([vx, vy])
        motion_norm = np.linalg.norm(motion_vec)
        if motion_norm > 1e-5:
            motion_vec /= motion_norm
            detection_vec = detection - last_pos
            det_norm = np.linalg.norm(detection_vec)
            if det_norm > 1e-5:
                detection_vec /= det_norm
                cos_theta = np.clip(np.dot(motion_vec, detection_vec), -1.0, 1.0)
                angle_penalty = (1 - cos_theta) * 50
    return dist + angle_penalty
# --- Visualization helper ---
def visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, num_prev, message=""):
    vis = frame.copy()
    active_tracks = [t for t in tracks.values() if t.active]
    active_ids = [t.tid for t in active_tracks]
   
    # Draw expected positions and directions for active tracks
    for tr in active_tracks:
        if tr.points:
            last_pos = np.array(tr.points[-1])
            if len(tr.points) > 1 and tr.pos_vel.vx is not None:
                vx, vy = tr.pos_vel.vx, tr.pos_vel.vy
                exp_color = (0, 255, 0)  # Green if using pos_vel
            else:
                vx, vy = tr.flow.vx, tr.flow.vy
                exp_color = (255, 255, 0)  # Cyan if using flow
            if vx is None:
                expected = last_pos
                exp_color = (0, 0, 255)  # Red if no velocity
            else:
                expected = last_pos + np.array([vx, vy])
            # Draw arrow from last_pos to expected if velocity exists
            if vx is not None:
                cv2.arrowedLine(vis, tuple(last_pos.astype(int)), tuple(expected.astype(int)), exp_color, 2, tipLength=0.3)
            # Draw circle at expected
            cv2.circle(vis, tuple(expected.astype(int)), 5, exp_color, 2)
            cv2.putText(vis, f"Exp:{tr.tid}", tuple(expected.astype(int) + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, exp_color, 1)
   
    # Draw tracks' trails
    for tr in active_tracks:
        if len(tr.points) < 1:
            continue
        pts = tr.points[-10:]
        for i in range(1, len(pts)):
            cv2.line(vis, tuple(pts[i-1].astype(int)), tuple(pts[i].astype(int)), tr.color, 2)
   
    # Draw bounding boxes and IDs
    for i, det in enumerate(all_detections[frame_idx]):
        x1, y1, x2, y2 = map(int, det[:4])
        color = (0, 255, 0)  # Green for detected
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        # Only draw center and ID if center exists and track_id is valid
        if i < len(centers):
            cx, cy = centers[i]
            track_id = assigned_tracks.get(i, None)
            if track_id is not None and track_id in tracks:
                tr = tracks[track_id]
                cv2.circle(vis, (int(cx), int(cy)), 3, tr.color, -1)
                cv2.putText(vis, f"ID:{tr.tid}", (int(cx + 5), int(cy + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tr.color, 1)
            elif track_id is not None:
                # Track ID assigned but not in tracks
                cv2.putText(vis, f"Invalid ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            # Indicate missing center
            cv2.putText(vis, "No center", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
   
    # Text overlays
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Active Tracks: {sorted(active_ids)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(vis, f"Curr Dets: {num_curr}, Prev Dets: {num_prev}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, f"Dets: {len(all_detections[frame_idx])}, Centers: {len(centers)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(vis, message, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
   
    cv2.imshow("Bee Tracking Debug", vis)
    key = cv2.waitKey(0) & 0xFF
    return key
# --- Missed detection handling as a function ---
def handle_missed_detection(tracks, centers, flows, next_track_id):
    active_tracks = [t for t in tracks.values() if t.active]
    temp_assignments = {j: set() for j in range(len(centers))}  # det_idx -> set of possible track tids
    unassigned_tracks = set(t.tid for t in active_tracks)
   
    # Step 1: For each track, find if there's a detection with low cost
    for tr in active_tracks:
        min_cost = np.inf
        best_det_idx = None
        thresh = 300 if len(tr.points) < 2 else 150
        for j, c in enumerate(centers):
            cost = compute_directional_cost(tr, c)
            if cost < min_cost:
                min_cost = cost
                best_det_idx = j
        if min_cost <= thresh:
            temp_assignments[best_det_idx].add(tr.tid)
   
    # Step 2: Resolve multiple assignments by choosing lowest cost for each det
    assigned_tracks = {}  # det_idx -> tid
    for det_idx, tids in temp_assignments.items():
        if len(tids) > 0:
            costs = {tid: compute_directional_cost(tracks[tid], centers[det_idx]) for tid in tids}
            best_tid = min(costs, key=costs.get)
            assigned_tracks[det_idx] = best_tid
            unassigned_tracks.discard(best_tid)
   
    # Step 3: Create new tracks for unassigned detections
    new_tracks = []
    unassigned_dets = set(range(len(centers))) - set(assigned_tracks.keys())
    for det_idx in unassigned_dets:
        color = tuple(random.randint(0, 255) for _ in range(3))
        new_tr = Track(next_track_id, color)
        new_tr.points.append(centers[det_idx])
        new_tr.flow.update(flows[det_idx] if flows and det_idx < len(flows) else None)
        new_tracks.append(new_tr)
        next_track_id += 1
   
    # Step 4: Mark unassigned tracks as pending deactivation
    for tid in unassigned_tracks:
        tracks[tid].pending_deactivation = True
   
    return assigned_tracks, new_tracks, next_track_id
# --- Main tracking function ---
def track_bees_stepwise(video_path, flow_folder_fwd, flow_folder_bwd, yolo_json_path):
    with open(yolo_json_path, "r") as f:
        all_detections = json.load(f)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Preload all frames, flows, detections
    frames = []
    centers_list = []
    flows_list = []
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {frame_idx}")
            frames.append(np.zeros((480, 640, 3), dtype=np.uint8))  # Placeholder frame
            centers_list.append([])
            flows_list.append([])
            continue
        flow_fwd_path = os.path.join(flow_folder_fwd, f"flow_fwd_{frame_idx:06d}.npy")
        flow_bwd_path = os.path.join(flow_folder_bwd, f"flow_bwd_{frame_idx:06d}.npy")
        centers = []
        flows = []
        if os.path.exists(flow_fwd_path) and os.path.exists(flow_bwd_path):
            flow_fwd = np.load(flow_fwd_path)
            flow_bwd = np.load(flow_bwd_path)
            valid_mask = forward_backward_mask(flow_fwd, flow_bwd)
            for det in all_detections[frame_idx]:
                x1, y1, x2, y2 = map(int, det[:4])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                med_flow = median_flow_in_bbox(flow_fwd, valid_mask, (x1, y1, x2, y2))
                centers.append(np.array([cx, cy]))
                flows.append(med_flow)
        else:
            print(f"Warning: Flow files missing for frame {frame_idx}")
            # Fallback: Create centers without flow
            for det in all_detections[frame_idx]:
                x1, y1, x2, y2 = map(int, det[:4])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centers.append(np.array([cx, cy]))
                flows.append(None)  # No flow data
        frames.append(frame.copy())
        centers_list.append(centers)
        flows_list.append(flows)
    cap.release()
    # Start tracking
    tracks = {}  # tid -> Track
    next_track_id = 0
    prev_num_dets = 0
    frame_idx = 0
    pending_missed_level = 0  # Flag to indicate if we're in a missed detection pass (1: second, 2: third, etc.)
    max_missed_passes = 3  # Base number of missed detection passes
    prev_pending_tids = set()  # Track IDs pending deactivation from previous pass
    while frame_idx < total_frames:
        frame = frames[frame_idx]
        centers = centers_list[frame_idx]
        flows = flows_list[frame_idx]
        num_curr = len(centers)
        active_tracks_list = [t for t in tracks.values() if t.active]
        message = f"Starting frame {frame_idx}. Curr dets: {num_curr}, Prev dets: {prev_num_dets}"
        key = visualize_frame(frame, all_detections, frame_idx, tracks, centers, {}, num_curr, prev_num_dets, message)
        if key == ord('q'):
            break
        elif key == 81:  # left arrow
            frame_idx = max(frame_idx - 1, 0)
            pending_missed_level = 0  # Reset missed detection state on rewind
            max_missed_passes = 3  # Reset max passes
            prev_pending_tids = set()
            continue  # No saving states, just rewind
        assigned_tracks = {}  # det_idx -> tid
        new_tracks = []
        if pending_missed_level > 0:
            # Additional pass of missed detection
            message = f"Missed detection pass {pending_missed_level + 1}."
            visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
            assigned_tracks, new_tracks, next_track_id = handle_missed_detection(tracks, centers, flows, next_track_id)
            message = f"After missed detection pass {pending_missed_level + 1}."
            visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
            # Add new tracks
            for new_tr in new_tracks:
                tracks[new_tr.tid] = new_tr
            # Apply assignments
            for det_idx, tid in assigned_tracks.items():
                if tid in tracks:
                    tr = tracks[tid]
                    tr.points.append(centers[det_idx])
                    tr.flow.update(flows[det_idx] if flows and det_idx < len(flows) else None)
                    if len(tr.points) > 1:
                        delta = tr.points[-1] - tr.points[-2]
                        tr.pos_vel.update((float(delta[0]), float(delta[1])))
                    tr.pending_deactivation = False
                else:
                    print(f"Warning: Track ID {tid} in assigned_tracks but not in tracks at frame {frame_idx}")
            # Get current pending deactivation tracks
            current_pending_tids = set(tid for tid, tr in tracks.items() if tr.active and tr.pending_deactivation)
            # Check if new tracks were added to pending deactivation since the first pass
            if pending_missed_level == 1:  # After second pass (first additional pass)
                prev_pending_tids = current_pending_tids.copy()
            elif pending_missed_level >= 2:  # After third pass or beyond
                new_pending = current_pending_tids - prev_pending_tids
                if new_pending:
                    max_missed_passes += 2  # Extend by two more passes if new tracks are pending
                    message = f"New tracks pending deactivation {new_pending}. Extending passes to {max_missed_passes}."
                    visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
                prev_pending_tids = current_pending_tids.copy()
            # Check for further pending deactivation
            any_pending = any(tr.pending_deactivation for tr in tracks.values() if tr.active)
            if any_pending and pending_missed_level < max_missed_passes:
                pending_missed_level += 1
                prev_num_dets = num_curr
                frame_idx += 1
                message = f"End of frame {frame_idx-1} processing. Moving to next frame for missed detection."
                visualize_frame(frame, all_detections, frame_idx-1, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
                continue
            # Deactivate tracks still pending
            for tr in list(tracks.values()):
                if tr.pending_deactivation:
                    tr.active = False
                    message = f"Deactivated track {tr.tid}"
                    visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
            # Reset pending_missed_level and proceed to next frame with normal logic
            pending_missed_level = 0
            max_missed_passes = 3  # Reset max passes
            prev_pending_tids = set()
            prev_num_dets = num_curr
            frame_idx += 1
            message = f"End of frame {frame_idx-1} processing (missed detection pass)."
            visualize_frame(frame, all_detections, frame_idx-1, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
            continue
        # Normal processing
        diff = num_curr - prev_num_dets
        if diff == 0:
            message = "Equal detections case. Assuming same tracks."
            visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
            if len(active_tracks_list) == 0 or num_curr == 0:
                # No tracks or no dets, create new tracks for all detections if detections exist
                for det_idx in range(num_curr):
                    color = tuple(random.randint(0, 255) for _ in range(3))
                    new_tr = Track(next_track_id, color)
                    new_tr.points.append(centers[det_idx])
                    new_tr.flow.update(flows[det_idx] if flows and det_idx < len(flows) else None)
                    new_tracks.append(new_tr)
                    assigned_tracks[det_idx] = new_tr.tid
                    tracks[new_tr.tid] = new_tr  # Add immediately
                    next_track_id += 1
                    message = f"Created new track {new_tr.tid} for det {det_idx} (no active tracks)"
                    visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
            else:
                # Cost optimization assignment
                cost_matrix = np.full((len(active_tracks_list), num_curr), np.inf)
                for i, tr in enumerate(active_tracks_list):
                    for j, c in enumerate(centers):
                        cost_matrix[i, j] = compute_directional_cost(tr, c)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                high_cost_dets = []
                high_cost_info = []
                for r, c in zip(row_ind, col_ind):
                    cost = cost_matrix[r, c]
                    if cost < np.inf:
                        tr = active_tracks_list[r]
                        thresh = 250 if len(tr.points) < 2 else 150
                        if cost > thresh:  # Fringe case 1: high cost -> new bee, missed detection
                            high_cost_dets.append(c)
                            high_cost_info.append(f"High cost {cost:.2f} (thresh {thresh}) for track {tr.tid} to det {c}")
                        else:
                            assigned_tracks[c] = tr.tid
                            message = f"Assigned det {c} to track {tr.tid} with cost {cost:.2f}"
                            visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
               
                if high_cost_dets:
                    message = "Fringe case 1: High cost assignments detected. Switching to missed detection case. " + "; ".join(high_cost_info)
                    visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
                    # Proceed to missed detection case
                    assigned_tracks, new_tracks, next_track_id = handle_missed_detection(tracks, centers, flows, next_track_id)
                    message = "After missed detection handling."
                    visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
        elif diff > 0:
            message = f"More detections case ({diff} more). Assuming {diff} new bees."
            visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
           
            # Check for false splitting positives
            to_remove = set()
            for i in range(num_curr):
                for j in range(i + 1, num_curr):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < 20:  # Arbitrary threshold for "very close"
                        # Assume false split, remove one (e.g., the second)
                        to_remove.add(j)
            if to_remove:
                message = f"Fringe case 2: False splitting detected. Removing {len(to_remove)} dets."
                visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
                # Remove and rerun the frame logic
                centers = [c for idx, c in enumerate(centers) if idx not in to_remove]
                flows = [f for idx, f in enumerate(flows) if idx not in to_remove]
                all_detections[frame_idx] = [d for idx, d in enumerate(all_detections[frame_idx]) if idx not in to_remove]
                centers_list[frame_idx] = centers
                flows_list[frame_idx] = flows
                num_curr = len(centers)
                diff = num_curr - prev_num_dets
                # Since we modified, continue to reprocess this frame
                continue
           
            # Assign to existing tracks using cost optimization
            if len(active_tracks_list) > 0 and num_curr > 0:
                cost_matrix = np.full((len(active_tracks_list), num_curr), np.inf)
                for i, tr in enumerate(active_tracks_list):
                    for j, c in enumerate(centers):
                        cost_matrix[i, j] = compute_directional_cost(tr, c)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    cost = cost_matrix[r, c]
                    tr = active_tracks_list[r]
                    thresh = 250 if len(tr.points) < 2 else 150
                    if cost < thresh:
                        assigned_tracks[c] = tr.tid
                        message = f"Assigned det {c} to track {tr.tid} with cost {cost:.2f}"
                        visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
           
            # Create new tracks for remaining
            unassigned_dets = set(range(num_curr)) - set(assigned_tracks.keys())
            for det_idx in unassigned_dets:
                color = tuple(random.randint(0, 255) for _ in range(3))
                new_tr = Track(next_track_id, color)
                new_tr.points.append(centers[det_idx])
                new_tr.flow.update(flows[det_idx] if flows and det_idx < len(flows) else None)
                new_tracks.append(new_tr)
                assigned_tracks[det_idx] = new_tr.tid
                tracks[new_tr.tid] = new_tr  # Add to tracks immediately
                next_track_id += 1
                message = f"Created new track {new_tr.tid} for det {det_idx}"
                visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
        else:  # diff < 0, fewer detections
            message = "Fewer detections case. Proceeding to missed detection case."
            visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
            assigned_tracks, new_tracks, next_track_id = handle_missed_detection(tracks, centers, flows, next_track_id)
            message = "After first missed detection pass."
            visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
           
            # Add new tracks
            for new_tr in new_tracks:
                tracks[new_tr.tid] = new_tr
           
            # Apply assignments
            for det_idx, tid in assigned_tracks.items():
                if tid in tracks:
                    tr = tracks[tid]
                    tr.points.append(centers[det_idx])
                    tr.flow.update(flows[det_idx] if flows and det_idx < len(flows) else None)
                    if len(tr.points) > 1:
                        delta = tr.points[-1] - tr.points[-2]
                        tr.pos_vel.update((float(delta[0]), float(delta[1])))
                    tr.pending_deactivation = False
                else:
                    print(f"Warning: Track ID {tid} in assigned_tracks but not in tracks at frame {frame_idx}")
            # Check for pending deactivation
            any_pending = any(tr.pending_deactivation for tr in tracks.values() if tr.active)
            if any_pending:
                # Move to next frame for additional missed detection pass
                pending_missed_level = 1
                prev_num_dets = num_curr
                prev_pending_tids = set(tid for tid, tr in tracks.items() if tr.active and tr.pending_deactivation)
                frame_idx += 1
                message = f"End of frame {frame_idx-1} processing. Moving to next frame for missed detection."
                visualize_frame(frame, all_detections, frame_idx-1, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
                continue
        # Apply assignments (for equal and more cases)
        for det_idx, tid in assigned_tracks.items():
            if tid in tracks:
                tr = tracks[tid]
                tr.points.append(centers[det_idx])
                tr.flow.update(flows[det_idx] if flows and det_idx < len(flows) else None)
                if len(tr.points) > 1:
                    delta = tr.points[-1] - tr.points[-2]
                    tr.pos_vel.update((float(delta[0]), float(delta[1])))
                tr.pending_deactivation = False
            else:
                print(f"Warning: Track ID {tid} in assigned_tracks but not in tracks at frame {frame_idx}")
        # Add new tracks
        for new_tr in new_tracks:
            if new_tr.tid not in tracks:
                tracks[new_tr.tid] = new_tr
        # Final visualization for frame
        message = f"End of frame {frame_idx} processing."
        visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, num_curr, prev_num_dets, message)
        prev_num_dets = num_curr
        frame_idx += 1
    cv2.destroyAllWindows()
if __name__ == "__main__":
    video_path = "data/videos/videoB.mp4"
    flow_folder_fwd = "data/raft_flows"
    flow_folder_bwd = "data/raft_flows"
    yolo_json_path = "data/yolo_detections/bboxes_orig.json"
    track_bees_stepwise(video_path, flow_folder_fwd, flow_folder_bwd, yolo_json_path)