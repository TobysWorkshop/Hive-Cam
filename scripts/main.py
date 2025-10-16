import sys
import os
from tkinter import Tk, filedialog

from generate_bboxes import bboxes_from_video
#from missed_count_box_tracker import track_bees_stepwise
from missed_count_polygon_tracker import track_bees_stepwise
from generate_entrances import generate_polygons


## -------------- VARIABLES ---------------- ##
YOLO_JSON_PATH = 'data/yolo_detections/bboxes.json'
OUTPUT_DIR = 'results'
## ----------------------------------------- ##


## Video file selection function ##
def select_video_file():
    """Open a file dialog to select a video file and return its path."""
    # Initialize Tkinter and hide the root window
    root = Tk()
    root.withdraw()  # Hide the main window
    
    # Open file dialog for video files
    file_path = filedialog.askopenfilename(
        title="Select a video file for input",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )
    
    # Destroy the Tkinter instance
    root.destroy()
    
    return file_path

## Pass the input video to the external bbox generating script and save the output
def get_bboxes(video_path):
    print("[INFO] Drawing bounding boxes... ", end='')
    bboxes_from_video(video_path)
    print("completed. Bboxes saved to data/yolo_detections/bboxes.json")


## ----------------------------------------------------------------------------------------------- ##
## ----------------------------------- Main Function  -------------------------------------------- ##
## ----------------------------------------------------------------------------------------------- ##

if __name__ == "__main__":
    # Prompt user to select a video file
    VIDEO_PATH = select_video_file()
    
    # Check if a file was selected
    if not VIDEO_PATH:
        print("[ERROR] No video file selected. Exiting run.")
        sys.exit(1)

    # Ensure the video file exists
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video file {VIDEO_PATH} does not exist. Exiting run.")
        sys.exit(1)

    ## Generate and save bboxes
    get_bboxes(VIDEO_PATH)

    ## Generate and save entrance polygons
    generate_polygons(VIDEO_PATH)

    # Begin the tracking function
    track_bees_stepwise(VIDEO_PATH, YOLO_JSON_PATH, OUTPUT_DIR, generate_video=False)