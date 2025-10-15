Here is a collection of all the key scripts that have been used throughout the project.

Currently, only three are used and therefore are of interest:
- main.py : the only script that needs to be called. This will run the other two scripts in the list autonomously.
- generate_bboxes.py : this script takes the input video selected in main.py and runs our yolo trained model on it to generate bounding boxes for each frame.
  These bboxes are then saved to data/yolo_detections/bboxes.json
- missed_count_box_tracker.py : this is the tracking script, that includes all the tracking logic and functions. It will use the video selected in main.py, and
  will load in the bboxes saved using generate_bboxes.py, and will track the bees across the video. The default result will be a graph of the bees entering and
  exiting over time, which will be saved in the results folder. The final counts will also be printed to the terminal.