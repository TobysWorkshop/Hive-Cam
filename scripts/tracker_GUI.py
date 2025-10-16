import numpy as np
import cv2
import json
import os
import random
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from average_polygons import compute_average_polygon
from PIL import Image, ImageTk
import tkinter as tk

OUTPUT_DIR = 'results'

# --- EMA helper for smoothing velocity ---
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

# --- Track class ---
class Track:
    def __init__(self, tid, color):
        self.tid = tid
        self.points = []
        self.pos_vel = EMA(alpha=0.4)
        self.color = color
        self.active = True
        self.miss_count = 0  # Count consecutive missed frames

# --- Cost function ---
def compute_directional_cost(track, detection, dt=1.0):
    if not track.points:
        return np.inf
    last_pos = np.array(track.points[-1])
    if len(track.points) > 1 and track.pos_vel.vx is not None:
        vx, vy = track.pos_vel.vx, track.pos_vel.vy
    else:
        vx, vy = None, None
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
def visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, in_count, out_count, hive_quad):
    vis = frame.copy()
    
    # Draw hive entrance quadrilateral
    for i in range(4):
        j = (i + 1) % 4
        cv2.line(vis, tuple(map(int, hive_quad[i])), tuple(map(int, hive_quad[j])), (0, 0, 255), 2)
    
    # Draw tracks for active tracks
    for tr in tracks.values():
        if tr.active and len(tr.points) > 1:
            for i in range(1, len(tr.points)):
                p1 = (int(tr.points[i-1][0]), int(tr.points[i-1][1]))
                p2 = (int(tr.points[i][0]), int(tr.points[i][1]))
                cv2.line(vis, p1, p2, tr.color, 1)
    
    # Draw bounding boxes and IDs
    for i, det in enumerate(all_detections[frame_idx]):
        x1, y1, x2, y2 = map(int, det[:4])
        color = (0, 255, 0)  # Green for detected
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if i < len(centers):
            cx, cy = centers[i]
            track_id = assigned_tracks.get(i, None)
            if track_id is not None and track_id in tracks:
                tr = tracks[track_id]
                cv2.circle(vis, (int(cx), int(cy)), 3, tr.color, -1)
                cv2.putText(vis, f"ID:{tr.tid}", (int(cx + 5), int(cy + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tr.color, 1)
            elif track_id is not None:
                cv2.putText(vis, f"Invalid ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            cv2.putText(vis, "No center", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Display bee counts
    cv2.putText(vis, f"Bees In: {in_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, f"Bees Out: {out_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return vis

# --- Missed detection handling ---
def handle_missed_detection(tracks, centers, next_track_id):
    active_tracks = [t for t in tracks.values() if t.active]
    temp_assignments = {j: set() for j in range(len(centers))}  # det_idx -> set of possible track tids
    unassigned_tracks = set(t.tid for t in active_tracks)
    
    # Step 1: Find detection with lowest cost for each track
    for tr in active_tracks:
        min_cost = np.inf
        best_det_idx = None
        thresh = 500 if tr.miss_count > 0 else (250 if len(tr.points) < 2 else 150)
        for j, c in enumerate(centers):
            cost = compute_directional_cost(tr, c)
            if cost < min_cost:
                min_cost = cost
                best_det_idx = j
        if min_cost <= thresh:
            temp_assignments[best_det_idx].add(tr.tid)
    
    # Step 2: Resolve multiple assignments
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
        new_tracks.append(new_tr)
        next_track_id += 1
    
    return assigned_tracks, new_tracks, next_track_id

# --- Function to define hive entrance quadrilateral using Tkinter ---
def define_hive_box(video_path, cap, tk_root=None):
    from PIL import Image, ImageTk

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None
    
    # Grab the first valid frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if ret:
            break
        frame_idx += 1
        if frame_idx >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            print("Error: No valid frames found in video")
            cap.release()
            return None
    cap.release()

    # Convert frame to PIL image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image)

    points = []

    # Create selection window
    window = tk.Toplevel(tk_root)
    window.title("Select Hive Entrance")
    canvas = tk.Canvas(window, width=photo.width(), height=photo.height())
    canvas.pack()
    canvas.photo = photo  # prevent garbage collection
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    info_label = tk.Label(window, text="Click four corners in order (e.g., clockwise).")
    info_label.pack(pady=5)

    def on_click(event):
        if len(points) < 4:
            points.append((event.x, event.y))
            canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill='red')
            if len(points) > 1:
                canvas.create_line(points[-2][0], points[-2][1], points[-1][0], points[-1][1], fill='red', width=2)
            if len(points) == 4:
                canvas.create_line(points[3][0], points[3][1], points[0][0], points[0][1], fill='red', width=2)
                info_label.config(text="All four corners selected. Press Confirm or Cancel.")

    canvas.bind("<Button-1>", on_click)

    result = tk.BooleanVar(value=False)

    button_frame = tk.Frame(window)
    button_frame.pack(pady=10)
    tk.Button(button_frame, text="Confirm", command=lambda: [result.set(True), window.destroy()]).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Cancel", command=lambda: [result.set(False), window.destroy()]).pack(side=tk.LEFT, padx=5)

    tk_root.wait_window(window)

    if result.get() and len(points) == 4:
        return [np.array(p, dtype=float) for p in points]
    else:
        print("Warning: Hive entrance quadrilateral not fully defined. Using default quadrilateral.")
        h, w = frame.shape[:2]
        return [np.array((w//4, h//4), dtype=float),
                np.array((3*w//4, h//4), dtype=float),
                np.array((3*w//4, 3*h//4), dtype=float),
                np.array((w//4, 3*h//4), dtype=float)]


# --- Function to confirm average polygon in Tkinter ---
def confirm_average_polygon(video_path, average_polygon, tk_root):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if ret:
            break
        frame_idx += 1
        if frame_idx >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            print("Error: No valid frames found in video")
            cap.release()
            return None
    vis = frame.copy()
    avg_quad = [np.array(p, dtype=float) for p in average_polygon]
    for i in range(4):
        j = (i + 1) % 4
        cv2.line(vis, tuple(map(int, avg_quad[i])), tuple(map(int, avg_quad[j])), (0, 0, 255), 2)
    cap.release()

    # Convert frame to PIL Image
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    h, w = vis.shape[:2]
    max_height = 400  # Limit display size
    if h > max_height:
        scale = max_height / h
        vis_rgb = cv2.resize(vis_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(vis_rgb)
    photo = ImageTk.PhotoImage(image)

    # Create confirmation window
    confirm_window = tk.Toplevel(tk_root)
    confirm_window.title("Confirm Hive Entrance")
    confirm_window.geometry(f"{photo.width()}x{photo.height()+80}")

    result = tk.BooleanVar(value=False)
    tk.Label(confirm_window, image=photo).pack(pady=10)
    tk.Label(confirm_window, text="Does this entrance zone look correct? If not, you will be asked to select your own.").pack()
    button_frame = tk.Frame(confirm_window)
    button_frame.pack(pady=10)
    tk.Button(button_frame, text="Yes", command=lambda: [result.set(True), confirm_window.destroy()]).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="No", command=lambda: [result.set(False), confirm_window.destroy()]).pack(side=tk.LEFT, padx=5)

    # Keep photo reference to prevent garbage collection
    confirm_window.photo = photo
    tk_root.wait_window(confirm_window)
    return result.get()

# --- Function to stabilize quadrilateral using feature matching ---
def stabilize_quad(prev_frame, curr_frame, quad, generate_video=True):
    if prev_frame is None or curr_frame is None:
        return quad  # No stabilization if frames are unavailable
    
    # Fast similarity check using subsampled pixels
    if not generate_video:
        h, w = prev_frame.shape[:2]
        sample_pixels = prev_frame[::10, ::10].flatten() - curr_frame[::10, ::10].flatten()
        diff = np.mean(sample_pixels ** 2)
        if diff < 15:
            #print(f"[DEBUG] stabilize_quad: Skipped frame, diff={diff:.2f}")
            return quad

    # Resize frames for stabilization when generate_video=False
    if not generate_video:
        scale_factor = 0.4  # Resize to ~50% resolution (e.g., 1420x800 -> ~710x400)
        prev_frame = cv2.resize(prev_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        curr_frame = cv2.resize(curr_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize AKAZE detector
    akaze = cv2.AKAZE_create()
    
    # Detect keypoints and descriptors
    kp1, des1 = akaze.detectAndCompute(prev_gray, None)
    kp2, des2 = akaze.detectAndCompute(curr_gray, None)

    # Reduce for memory-saving
    #kp1 = sorted(kp1, key=lambda x: x.response, reverse=True)[:300]
    #kp2 = sorted(kp2, key=lambda x: x.response, reverse=True)[:300]
    #des1 = des1[:300]
    #des2 = des2[:300]
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return quad  # Return unchanged if insufficient keypoints
    
    # Match descriptors using BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    
    if len(matches) < 4:
        return quad  # Need at least 4 matches for homography
    
    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:min(30, len(matches))]  # Limit to top 30 matches
    
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return quad  # Return unchanged if homography fails
    
    # Scale homography back to original resolution if resized
    if not generate_video:
        scale_matrix = np.array([[1.0 / scale_factor, 0, 0], [0, 1.0 / scale_factor, 0], [0, 0, 1]])
        H = scale_matrix @ H @ np.linalg.inv(scale_matrix)
    
    # Transform quadrilateral points
    quad_pts = np.float32(quad).reshape(-1, 1, 2)
    new_quad_pts = cv2.perspectiveTransform(quad_pts, H)
    new_quad = new_quad_pts.reshape(-1, 2).tolist()
    
    return [np.array(p, dtype=float) for p in new_quad]

# --- Function to check if a point is inside the quadrilateral ---
def is_inside_box(point, quad):
    x, y = point
    inside = False
    n = len(quad)
    for i in range(n):
        j = (i + 1) % n
        xi, yi = quad[i]
        xj, yj = quad[j]
        if ((yi > y) != (yj > y)):
            xinters = (y - yi) * (xj - xi) / (yj - yi) + xi
            if x < xinters:
                inside = not inside
    return inside

# --- Plotting function for bee counts ---
def plot_bee_counts(entry_timestamps, exit_timestamps, video_duration, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define time bins (1-second intervals)
    max_time = int(video_duration) + 1
    time_bins = np.arange(0, max_time + 1, 1)
    
    # Initialize cumulative counts
    in_counts = np.zeros(len(time_bins))
    out_counts = np.zeros(len(time_bins))
    
    # Calculate cumulative counts
    for t in entry_timestamps:
        bin_idx = min(int(t), max_time - 1)
        in_counts[bin_idx:] += 1
    for t in exit_timestamps:
        bin_idx = min(int(t), max_time - 1)
        out_counts[bin_idx:] += 1
    
    # Initialize histogram counts (non-cumulative)
    hist_in_counts, _ = np.histogram(entry_timestamps, bins=time_bins)
    hist_out_counts, _ = np.histogram(exit_timestamps, bins=time_bins)
    
    # Set seaborn style
    sns.set_style("darkgrid")
    
    # Plot 1: Cumulative line plot (original)
    plt.figure(figsize=(10, 6))
    plt.plot(time_bins, in_counts, label='Bees Entering', color='blue')
    plt.plot(time_bins, out_counts, label='Bees Exiting', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cumulative Count')
    max_count = int(np.max([in_counts, out_counts])) + 3
    plt.yticks(np.arange(0, max_count, 1))
    plt.ylim(0, max_count)
    plt.xlim(-2.5, max_time + 2.5)
    plt.title('Bee Entries and Exits Over Time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'bee_counts_line.png'))
    plt.close()
    
    # Plot 2: Binned histogram with cumulative lines
    plt.figure(figsize=(10, 6))
    # Plot histogram bars (slightly offset for visibility)
    bar_width = 0.45
    plt.bar(time_bins[:-1], hist_in_counts, width=bar_width, label='Bees Entering (Histogram)', color='blue', alpha=0.5)
    plt.bar(time_bins[:-1] + bar_width, hist_out_counts, width=bar_width, label='Bees Exiting (Histogram)', color='red', alpha=0.5)
    # Overlay cumulative lines
    plt.plot(time_bins, in_counts, label='Cumulative Bees Entering', color='blue', linestyle='--')
    plt.plot(time_bins, out_counts, label='Cumulative Bees Exiting', color='red', linestyle='--')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Count per Second')
    plt.yticks(np.arange(0, max_count, 1))
    plt.ylim(0, max_count)
    plt.xlim(-2.5, max_time + 2.5)
    plt.title('Bee Entries and Exits Over Time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'bee_counts_histogram.png'))
    plt.close()

# --- Main tracking function ---
def track_bees_stepwise(video_path, yolo_json_path, entrance_json_path, output_dir, generate_video=False, tk_root=None):
    # Compute average polygon
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
    average_polygon = compute_average_polygon(entrance_json_path)
    if average_polygon is None:
        print("Warning: Could not compute average polygon. Falling back to manual selection.")
        hive_quad = define_hive_box(video_path, cap, tk_root)
        if hive_quad is None:
            print("Error: Could not define hive entrance quadrilateral. Exiting.")
            cap.release()
            return
    else:
        confirmed = confirm_average_polygon(video_path, average_polygon, tk_root)
        if confirmed:
            hive_quad = [np.array(p, dtype=float) for p in average_polygon]
            print("[INFO] User confirmed average polygon.")
        else:
            print("[INFO] User rejected average polygon. Falling back to manual selection.")
            hive_quad = define_hive_box(video_path, cap, tk_root)
            if hive_quad is None:
                print("Error: Could not define hive entrance quadrilateral. Exiting.")
                cap.release()
                return
    
    hive_quad_current = hive_quad
    
    with open(yolo_json_path, "r") as f:
        all_detections = json.load(f)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / fps
    
    # Initialize video writer only if generating video
    if generate_video:
        output_path = os.path.join(output_dir, 'output_missed_count_tracker.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize counters and timestamp logs
    in_count = 0
    out_count = 0
    entry_timestamps = []
    exit_timestamps = []
    
    # Start tracking
    tracks = {}  # tid -> Track
    next_track_id = 0
    prev_num_dets = 0
    prev_frame = None
    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {frame_idx}")
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            centers = []
        else:
            # Compute centers for current frame
            centers = []
            for det in all_detections[frame_idx]:
                x1, y1, x2, y2 = map(int, det[:4])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centers.append(np.array([cx, cy]))
        
        # Stabilize hive quadrilateral
        if frame_idx > 0 and ret:
            hive_quad_current = stabilize_quad(prev_frame, frame, hive_quad_current, generate_video)
        
        num_curr = len(centers)
        active_tracks_list = [t for t in tracks.values() if t.active]
        diff = num_curr - prev_num_dets
        assigned_tracks = {}  # det_idx -> tid
        new_tracks = []
        more_new_tracks = []
        even_more_new_tracks = []
        if diff == 0:
            if len(active_tracks_list) == 0 or num_curr == 0:
                for det_idx in range(num_curr):
                    color = tuple(random.randint(0, 255) for _ in range(3))
                    new_tr = Track(next_track_id, color)
                    new_tr.points.append(centers[det_idx])
                    new_tracks.append(new_tr)
                    assigned_tracks[det_idx] = new_tr.tid
                    tracks[new_tr.tid] = new_tr
                    next_track_id += 1
            else:
                cost_matrix = np.full((len(active_tracks_list), num_curr), np.inf)
                for i, tr in enumerate(active_tracks_list):
                    for j, c in enumerate(centers):
                        cost_matrix[i, j] = compute_directional_cost(tr, c)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                high_cost_dets = []
                for r, c in zip(row_ind, col_ind):
                    cost = cost_matrix[r, c]
                    if cost < np.inf:
                        tr = active_tracks_list[r]
                        thresh = 500 if tr.miss_count > 0 else (250 if len(tr.points) < 2 else 150)
                        if cost > thresh:
                            high_cost_dets.append(c)
                        else:
                            assigned_tracks[c] = tr.tid
                
                if high_cost_dets:
                    assigned_tracks, new_tracks, next_track_id = handle_missed_detection(tracks, centers, next_track_id)
        elif diff > 0:
            to_remove = set()
            for i in range(num_curr):
                for j in range(i + 1, num_curr):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < 20:
                        to_remove.add(j)
            if to_remove:
                centers = [c for idx, c in enumerate(centers) if idx not in to_remove]
                all_detections[frame_idx] = [d for idx, d in enumerate(all_detections[frame_idx]) if idx not in to_remove]
                num_curr = len(centers)
                diff = num_curr - prev_num_dets
                continue
            
            if len(active_tracks_list) > 0 and num_curr > 0:
                cost_matrix = np.full((len(active_tracks_list), num_curr), np.inf)
                for i, tr in enumerate(active_tracks_list):
                    for j, c in enumerate(centers):
                        cost_matrix[i, j] = compute_directional_cost(tr, c)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    cost = cost_matrix[r, c]
                    tr = active_tracks_list[r]
                    thresh = 500 if tr.miss_count > 0 else (250 if len(tr.points) < 2 else 150)
                    if cost < thresh:
                        assigned_tracks[c] = tr.tid
            
            unassigned_dets = set(range(num_curr)) - set(assigned_tracks.keys())
            for det_idx in unassigned_dets:
                color = tuple(random.randint(0, 255) for _ in range(3))
                new_tr = Track(next_track_id, color)
                new_tr.points.append(centers[det_idx])
                new_tracks.append(new_tr)
                assigned_tracks[det_idx] = new_tr.tid
                tracks[new_tr.tid] = new_tr
                next_track_id += 1
        else:  # diff < 0
            assigned_tracks, new_tracks, next_track_id = handle_missed_detection(tracks, centers, next_track_id)
            
            for new_tr in new_tracks:
                tracks[new_tr.tid] = new_tr
        
        # Apply assignments
        for det_idx, tid in assigned_tracks.items():
            if tid in tracks:
                tr = tracks[tid]
                tr.points.append(centers[det_idx])
                if len(tr.points) > 1:
                    delta = tr.points[-1] - tr.points[-2]
                    tr.pos_vel.update((float(delta[0]), float(delta[1])))
                tr.miss_count = 0
        
        # Add new tracks
        all_new_tracks = new_tracks + more_new_tracks + even_more_new_tracks if diff >= 0 else new_tracks
        for new_tr in all_new_tracks:
            if new_tr.tid not in tracks:
                tracks[new_tr.tid] = new_tr
        
        # Update miss counts, deactivate, and count bees
        assigned_tids = set(assigned_tracks.values())
        for tid, tr in list(tracks.items()):
            if tr.active:
                if tid in assigned_tids:
                    tr.miss_count = 0
                else:
                    # Check if track is going IN and last point is inside the box
                    if len(tr.points) > 1:  # Ensure there's at least one point to check
                        last_point = tr.points[-1]
                        start_point = tr.points[0]
                        last_inside = is_inside_box(last_point, hive_quad_current)
                        start_inside = is_inside_box(start_point, hive_quad_current)
                        going_in = not start_inside and last_inside
                        if going_in and last_inside:
                            tr.active = False
                            if len(tr.points) > 2:
                                x_diff = last_point[0] - start_point[0]
                                if abs(x_diff) > 10:
                                    timestamp = frame_idx / fps
                                    in_count += 1
                                    entry_timestamps.append(timestamp)
                        else:
                            tr.miss_count += 1
                    else:
                        tr.miss_count += 1
                if tr.miss_count >= 3:
                    tr.active = False
                    if len(tr.points) > 2:
                        x_diff = tr.points[-1][0] - tr.points[0][0]
                        if abs(x_diff) > 10:
                            start_point = tr.points[0]
                            end_point = tr.points[-1]
                            start_inside = is_inside_box(start_point, hive_quad_current)
                            end_inside = is_inside_box(end_point, hive_quad_current)
                            timestamp = frame_idx / fps
                            if not start_inside and end_inside:
                                in_count += 1
                                entry_timestamps.append(timestamp)
                            elif start_inside and not end_inside:
                                out_count += 1
                                exit_timestamps.append(timestamp)
        
        # Visualize and write frame if generating video
        if generate_video:
            vis_frame = visualize_frame(frame, all_detections, frame_idx, tracks, centers, assigned_tracks, in_count, out_count, hive_quad_current)
            out.write(vis_frame)
        
        prev_num_dets = num_curr
        prev_frame = frame.copy() if ret else None
        frame_idx += 1
        yield (frame_idx / total_frames * 100), None  # Yield progress percentage
    
    # Release video writer and capture
    if generate_video:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    # Plot bee counts
    plot_bee_counts(entry_timestamps, exit_timestamps, video_duration, output_dir)
    
    result = f"[RESULT] Final Bee Counts: In = {in_count}, Out = {out_count}"

    # Print final bee counts
    print(result)

    yield 100.0, result

if __name__ == "__main__":
    video_path = "data/videos/videoB.mp4"
    yolo_json_path = "data/yolo_detections/bboxes.json"
    entrance_json_path = "data/yolo_detections/polygons.json"
    output_dir = "results"

    #import cProfile
    #import pstats
    #profile_output_file = "profiling_output.txt"
    #profiler = cProfile.Profile()
    #profiler.enable()
    track_bees_stepwise(video_path, yolo_json_path, entrance_json_path, output_dir, generate_video=False)
    #profiler.disable()
    #with open(profile_output_file, 'w') as f:
    #    ps = pstats.Stats(profiler, stream=f)
    #    ps.sort_stats('cumulative')  # Sort by cumulative time
    #    ps.print_stats()  # Write detailed stats to file

    #print(f"Profiling results saved to {profile_output_file}")