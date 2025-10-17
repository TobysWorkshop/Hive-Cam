import cv2
from ultralytics import YOLO
import numpy as np
import torch
import json
import os

#OUTPUT_DIR = "data/yolo_detections"
#WEIGHTS = 'weights/entrance.pt'
SAMPLE_RATE = 20 # Takes a detection every this number of frames. 

def sort_quad_points(pts):
    pts = pts.reshape(4, 2)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    sort_idx = np.argsort(angles)
    sorted_pts = pts[sort_idx]
    return sorted_pts.reshape(4, 1, 2).astype(np.int32)

def force_parallelogram(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    errors = []
    corrected = []
    
    # Correct p3: p3' = p0 + p2 - p1
    p3_prime = pts[0] + pts[2] - pts[1]
    err = np.linalg.norm(p3_prime - pts[3])
    errors.append(err)
    corrected.append(np.array([pts[0], pts[1], pts[2], p3_prime]))
    
    # Correct p0: p0' = p1 + p3 - p2
    p0_prime = pts[1] + pts[3] - pts[2]
    err = np.linalg.norm(p0_prime - pts[0])
    errors.append(err)
    corrected.append(np.array([p0_prime, pts[1], pts[2], pts[3]]))
    
    # Correct p1: p1' = p0 + p2 - p3
    p1_prime = pts[0] + pts[2] - pts[3]
    err = np.linalg.norm(p1_prime - pts[1])
    errors.append(err)
    corrected.append(np.array([pts[0], p1_prime, pts[2], pts[3]]))
    
    # Correct p2: p2' = p1 + p3 - p0
    p2_prime = pts[1] + pts[3] - pts[0]
    err = np.linalg.norm(p2_prime - pts[2])
    errors.append(err)
    corrected.append(np.array([pts[0], pts[1], p2_prime, pts[3]]))
    
    min_idx = np.argmin(errors)
    best_pts = corrected[min_idx]
    return best_pts.reshape(4, 1, 2).astype(np.int32)

def expand_quad(pts, scale=1.05):
    pts = pts.reshape(4, 2).astype(np.float32)
    center = np.mean(pts, axis=0)
    pts = center + scale * (pts - center)
    return pts.reshape(4, 1, 2).astype(np.int32)



# Create a safe output directory next to the exe or script
out_dir = os.path.join(os.getcwd(), "output")
os.makedirs(out_dir, exist_ok=True)


def generate_polygons(video_path, weights_path):
    # Load YOLO segmentation model with custom weights
    model = YOLO(weights_path)
    
    # Dictionary for per-object buffers (list of polys for averaging)
    buffers = {}
    buffer_size = 5  # Number of frames for smoothing
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file for polygon detection.")
        return
    
    # Data structure to hold frame information
    frame_data = {}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            #print(f"Error reading frame {frame_idx}")
            break
        
        # Initialize entry for this frame
        frame_data[str(frame_idx)] = []
        
        # Run inference with tracking
        results = model.track(frame, persist=True, verbose=False)
        
        # Process detections
        if results[0].masks:  # If there are masks
            ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None
            for i in range(len(results[0].boxes)):
                conf = results[0].boxes.conf[i]  # Confidence score
                if conf > 0.5:  # Confidence threshold
                    # Get the polygon points
                    poly = results[0].masks.xy[i].astype(np.int32)
                    if len(poly) < 4:
                        continue  # Too few points
                    
                    # Reshape to OpenCV contour format (N,1,2)
                    contour = poly.reshape(-1, 1, 2)
                    
                    # Approximate to quadrilateral
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.005 * perimeter  # Start with small epsilon
                    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                    
                    max_iterations = SAMPLE_RATE
                    iter_count = 0
                    while len(approx) > 4 and iter_count < max_iterations:
                        epsilon *= 1.2  # Increase epsilon
                        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                        iter_count += 1
                    
                    if len(approx) != 4:
                        #print(f"Warning: Could not approximate to exactly 4 sides (got {len(approx)}). Using approximation.")
                        if len(approx) < 4:
                            continue
                    
                    # Sort points
                    approx = sort_quad_points(approx)
                    
                    # Force to parallelogram
                    approx = force_parallelogram(approx)
                    
                    # Expand to make generous
                    approx = expand_quad(approx, scale=1.05)
                    
                    # Smoothing with buffer
                    approx_flat = approx.reshape(4, 2).astype(np.float32)
                    obj_id = ids[i] if ids is not None else i  # Use track ID if available, else index
                    if obj_id not in buffers:
                        buffers[obj_id] = []
                    buffers[obj_id].append(approx_flat)
                    if len(buffers[obj_id]) > buffer_size:
                        buffers[obj_id].pop(0)
                    
                    # Compute smoothed poly
                    smoothed_pts = np.mean(buffers[obj_id], axis=0)
                    
                    # Save the points for this polygon
                    points_list = smoothed_pts.tolist()  # Convert to list of lists [[x1,y1], [x2,y2], ...]
                    frame_data[str(frame_idx)].append(points_list)
        
        frame_idx += SAMPLE_RATE
        yield (frame_idx / total_frames * 100)
    
    # Release resources
    cap.release()
    
    # Save to JSON file
    json_path = os.path.join(out_dir, 'polygons.json')
    with open(json_path, 'w') as f:
        json.dump(frame_data, f, indent=4)
    #print(f"[INFO] Saved entrance polygons to {json_path}")

    yield 100.0

#if __name__ == "__main__":
#    generate_polygons('data/videos/videoB.mp4')