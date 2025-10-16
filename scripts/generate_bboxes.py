# --- IMPORTING DEPENDENCIES --- #
print("[INFO] Importing dependencies... ", end="")

import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

print("completed.")


# --- CONFIGURATION --- #
MODEL_PATH = 'weights/2000-lvl-model.pt'
CONF_THRES = 0.25
OUTPUT_DIR = "data"

# --- INITIALISE MODELS --- #
print("[INFO] Initialising models... ", end="")
model = YOLO(MODEL_PATH).to("cuda")

print("completed.")

# --- YOLO BOUNDING BOXES --- #
def bboxes_from_video(
    video_path,
    out_dir=os.path.join(OUTPUT_DIR, "yolo_detections"),
):
    """
    Runs YOLO on the input video and saves detections.
    """

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    bboxes_orig = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRES, verbose=False)
        frame_boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                # Original
                frame_boxes.append([x1, y1, x2, y2, conf, class_id])

        bboxes_orig.append(frame_boxes)

        frame_idx += 1

    # Save the two JSONs
    path = os.path.join(out_dir, "bboxes.json")

    with open(path, "w") as f:
        json.dump(bboxes_orig, f)

    cap.release()


# --- MAIN PIPELINE --- #
if __name__ == "__main__":
    print("[INFO] Drawing bounding boxes... ", end='')
    bboxes_from_video()
    print("completed.")

    print("[INFO] Processing complete. Bboxes saved to data/yolo_detections/bboxes.json")
