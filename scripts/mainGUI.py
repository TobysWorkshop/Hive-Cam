import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
from generate_bboxesGUI import bboxes_from_video
from generate_entrancesGUI import generate_polygons
from tracker_GUI import track_bees_stepwise
import threading
from io import StringIO
from PIL import Image, ImageTk
import re

# Ensure directories exist
#os.makedirs('data/yolo_detections', exist_ok=True)
#os.makedirs(OUTPUT_DIR, exist_ok=True)

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS  # temporary folder for PyInstaller
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Create a safe output directory next to the exe or script
out_dir = os.path.join(os.getcwd(), "output")
os.makedirs(out_dir, exist_ok=True)

# Define default paths
YOLO_JSON_PATH = os.path.join(out_dir, 'yolo_detections/bboxes.json')
POLYGON_JSON_PATH = os.path.join(out_dir, 'polygons.json')
OUTPUT_DIR = out_dir
weights_path = resource_path("weights/2000-lvl-model.pt")


# Custom tooltip class
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT, background="#F5F5F5", relief=tk.SOLID, borderwidth=1, font=("Segoe UI", 10), foreground="#000000")
        label.pack()

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

# GUI class
class BeeTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hive-Cam")
        self.root.geometry("1400x700")  # Wider window
        self.root.configure(bg="#F5F5F5")  # Very light gray background
        self.root.resizable(False, False)
        icon = resource_path("bee.ico")
        self.root.iconbitmap(icon)

        # Initialize variables first
        self.video_path = tk.StringVar()
        self.generate_video = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Ready")
        self.results_text = tk.StringVar(value="")
        self.detection_progress = tk.DoubleVar(value=0.0)
        self.tracking_progress = tk.DoubleVar(value=0.0)
        self.histogram_photo = None
        self.is_running = False

        # Apply modern style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.TButton", font=("Segoe UI", 11), padding=8, background="#FFB300", foreground="#000000", borderwidth=0)
        style.map("Custom.TButton", background=[("active", "#FFA000")])
        style.configure("TLabel", font=("Segoe UI", 11), foreground="#000000")  # No background
        style.configure("TCheckbutton", font=("Segoe UI", 11), foreground="#000000")
        style.configure("TEntry", font=("Segoe UI", 11), background="#E8E8E8", fieldbackground="#E8E8E8", highlightthickness=0, borderwidth=0)
        style.configure("Custom.Horizontal.TProgressbar", thickness=20, troughcolor="#E8E8E8", background="#FFB300", borderwidth=0)
        style.configure("TFrame", background="#F5F5F5")
        style.configure("TLabelframe", background="#E8E8E8", relief="flat", borderwidth=0, padding=(5, 0, 0, 0))  # Padding for section titles
        style.configure("TLabelframe.Label", font=("Segoe UI", 12, "bold"), foreground="#000000", background="#E8E8E8")

        # Main container
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Main title (above columns)
        ttk.Label(main_frame, text="Hive-Cam", font=("Segoe UI", 24, "bold"), foreground="#000000").pack(pady=(0, 20))

        # Columns container
        columns_frame = ttk.Frame(main_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True)

        # Left column: Inputs, Progress, Status
        left_frame = ttk.Frame(columns_frame, width=600)  # Same width
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_frame.pack_propagate(False)

        # Input Selection Frame
        input_frame = ttk.LabelFrame(left_frame, text="Input Selection", padding=12)
        input_frame.pack(fill=tk.X, pady=10)

        video_frame = ttk.Frame(input_frame)
        video_frame.pack(fill=tk.X, pady=5)
        ttk.Label(video_frame, text="Video File:").pack(side=tk.LEFT)
        self.video_entry = ttk.Entry(video_frame, textvariable=self.video_path, width=40, style="TEntry")
        self.video_entry.pack(side=tk.LEFT, padx=5)
        ToolTip(self.video_entry, "Select a video file (.mp4, .avi, .mov, .mkv)")
        browse_button = ttk.Button(video_frame, text="Browse", command=self.browse_video, style="Custom.TButton")
        browse_button.pack(side=tk.LEFT)
        ToolTip(browse_button, "Choose a video file to process")

        ttk.Checkbutton(input_frame, text="Generate Video Output", variable=self.generate_video).pack(pady=5)
        ToolTip(input_frame.winfo_children()[-1], "Check to generate an output video with tracking visualizations")

        # Progress Frame
        progress_frame = ttk.LabelFrame(left_frame, text="Progress", padding=12)
        progress_frame.pack(fill=tk.X, pady=10)

        detection_progress_frame = ttk.Frame(progress_frame)
        detection_progress_frame.pack(fill=tk.X, pady=5)
        ttk.Label(detection_progress_frame, text="Detection:").pack(side=tk.LEFT)
        self.detection_progress_bar = ttk.Progressbar(detection_progress_frame, variable=self.detection_progress, maximum=100, style="Custom.Horizontal.TProgressbar")
        self.detection_progress_bar.pack(fill=tk.X, side=tk.LEFT, padx=5, expand=True)
        ToolTip(self.detection_progress_bar, "Progress of bounding box and polygon detection")

        tracking_progress_frame = ttk.Frame(progress_frame)
        tracking_progress_frame.pack(fill=tk.X, pady=5)
        ttk.Label(tracking_progress_frame, text="Tracking:").pack(side=tk.LEFT)
        self.tracking_progress_bar = ttk.Progressbar(tracking_progress_frame, variable=self.tracking_progress, maximum=100, style="Custom.Horizontal.TProgressbar")
        self.tracking_progress_bar.pack(fill=tk.X, side=tk.LEFT, padx=5, expand=True)
        ToolTip(self.tracking_progress_bar, "Progress of bee tracking")

        # Buttons Frame
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=15)
        self.run_button = ttk.Button(button_frame, text="Run Tracking", command=self.run_tracking, style="Custom.TButton")
        self.run_button.pack(side=tk.LEFT, padx=5)
        ToolTip(self.run_button, "Start the bee tracking process")
        ttk.Button(button_frame, text="Clear", command=self.clear_inputs, style="Custom.TButton").pack(side=tk.LEFT, padx=5)
        ToolTip(button_frame.winfo_children()[-1], "Reset inputs and results")
        ttk.Button(button_frame, text="Exit", command=self.exit_app, style="Custom.TButton").pack(side=tk.LEFT, padx=5)
        ToolTip(button_frame.winfo_children()[-1], "Close the application")

        # Status Frame
        status_frame = ttk.LabelFrame(left_frame, text="Status", padding=12)
        status_frame.pack(fill=tk.X, pady=10)
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_text, wraplength=500).pack(side=tk.LEFT)
        ToolTip(status_frame.winfo_children()[-1], "Current processing status")

        # Right column: Results and Histogram
        right_frame = ttk.Frame(columns_frame, width=700)  # Wider for histogram
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # Results Frame
        results_frame = ttk.LabelFrame(right_frame, text="Results", padding=12)
        results_frame.pack(fill=tk.BOTH, pady=10, padx=10)
        ttk.Label(results_frame, text="Counts:", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(results_frame, textvariable=self.results_text, wraplength=600, font=("Segoe UI", 11)).pack(anchor="w")
        ToolTip(results_frame.winfo_children()[-1], "Number of bees entering and exiting the hive")

        # Histogram Image
        self.histogram_label = ttk.Label(right_frame, borderwidth=2, relief="groove")
        self.histogram_label.pack(pady=15, padx=10)
        ToolTip(self.histogram_label, "Histogram of bee entries and exits over time")

    def browse_video(self):
        file_path = filedialog.askopenfilename(
            title="Select a video file for input",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.video_path.set(file_path)
            self.status_text.set("Video selected.")

    def get_bboxes(self, video_path, weights_path):
        """Generate and save bounding boxes."""
        self.status_text.set("Generating bounding boxes...")
        self.root.update()
        print("[INFO] Drawing bounding boxes... ", end='')
        try:
            for progress in bboxes_from_video(video_path, weights_path):
                self.root.after(0, lambda p=progress: self.detection_progress.set(p))
            print("completed. Bboxes saved to data/yolo_detections/bboxes.json")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to generate bboxes: {str(e)}")
            return f"Error generating bboxes: {str(e)}"

    def get_entrance_polygons(self, video_path, weights_path):
        """Generate and save entrance polygons."""
        self.status_text.set("Isolating hive entrances...")
        self.root.update()
        print("[INFO] Isolating hive entrances... ", end='')
        try:
            for progress in generate_polygons(video_path, weights_path):
                self.root.after(0, lambda p=progress: self.detection_progress.set(p))
            print("completed. Polygons saved to data/yolo_detections/polygons.json")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to generate polygons: {str(e)}")
            return f"Error generating polygons: {str(e)}"

    def run_tracking(self):
        if self.is_running:
            return
        self.is_running = True
        self.run_button.configure(state="disabled")

        # Clear previous status, results, progress, and image
        self.status_text.set("")
        self.results_text.set("")
        self.detection_progress.set(0.0)
        self.tracking_progress.set(0.0)
        self.histogram_label.configure(image='')

        # Get inputs
        video_path = self.video_path.get()
        generate_video = self.generate_video.get()

        # Validate inputs
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file.")
            self.status_text.set("No valid video file selected.")
            self.is_running = False
            self.run_button.configure(state="normal")
            return

        # Run processing in a separate thread
        threading.Thread(
            target=self.process_tracking,
            args=(video_path, generate_video),
            daemon=True
        ).start()

    def process_tracking(self, video_path, generate_video):
        try:
            self.root.after(0, lambda: self.status_text.set("Starting processing..."))
            print("[INFO] Video loaded successfully.")

            # Generate bboxes
            bbox_result = self.get_bboxes(video_path, weights_path)
            if bbox_result is not True:
                self.root.after(0, lambda: messagebox.showerror("Error", bbox_result))
                self.root.after(0, lambda: self.status_text.set("Failed to generate bounding boxes."))
                self.root.after(0, lambda: self.set_running_false())
                return

            # Generate entrance polygons
            polygon_result = self.get_entrance_polygons(video_path, weights_path)
            if polygon_result is not True:
                self.root.after(0, lambda: messagebox.showerror("Error", polygon_result))
                self.root.after(0, lambda: self.status_text.set("Failed to generate entrance polygons."))
                self.root.after(0, lambda: self.set_running_false())
                return

            # Verify generated files
            if not os.path.exists(YOLO_JSON_PATH):
                self.root.after(0, lambda: messagebox.showerror("Error", "Bounding box JSON file was not generated."))
                self.root.after(0, lambda: self.status_text.set("Missing bboxes.json."))
                self.root.after(0, lambda: self.set_running_false())
                return
            if not os.path.exists(POLYGON_JSON_PATH):
                self.root.after(0, lambda: messagebox.showerror("Error", "Entrance polygon JSON file was not generated."))
                self.root.after(0, lambda: self.status_text.set("Missing polygons.json."))
                self.root.after(0, lambda: self.set_running_false())
                return

            # Run tracking with progress updates
            self.root.after(0, lambda: self.status_text.set("Tracking bees..."))
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            for progress, result in track_bees_stepwise(video_path, YOLO_JSON_PATH, POLYGON_JSON_PATH, OUTPUT_DIR, generate_video, self.root):
                if isinstance(result, str) and "[RESULT] Final Bee Counts" in result:
                    # Parse counts from result
                    match = re.search(r"In = (\d+), Out = (\d+)", result)
                    if match:
                        in_count, out_count = map(int, match.groups())
                        result_text = f"Bees entering the hive: {in_count}\nBees exiting the hive: {out_count}"
                        self.root.after(0, lambda r=result_text: self.results_text.set(r))
                self.root.after(0, lambda p=progress: self.tracking_progress.set(p))
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            # Load and display histogram
            try:
                image = Image.open(os.path.join(OUTPUT_DIR, 'bee_counts_histogram.png'))
                image = image.resize((600, 450), Image.Resampling.LANCZOS)
                self.histogram_photo = ImageTk.PhotoImage(image)
                self.root.after(0, lambda: self.histogram_label.configure(image=self.histogram_photo))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load histogram: {str(e)}"))

            self.root.after(0, lambda: self.status_text.set("Tracking completed. Check output directory for plots."))
            self.root.after(0, lambda: messagebox.showinfo("Success", "Tracking complete! Check results in output directory."))
            self.root.after(0, lambda: self.set_running_false())
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Tracking error: {str(e)}"))
            self.root.after(0, lambda: self.status_text.set("Tracking failed."))
            self.root.after(0, lambda: self.results_text.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: self.set_running_false())

    def clear_inputs(self):
        """Reset inputs and results."""
        self.video_path.set("")
        self.generate_video.set(False)
        self.status_text.set("Ready")
        self.results_text.set("")
        self.detection_progress.set(0.0)
        self.tracking_progress.set(0.0)
        self.histogram_label.configure(image='')
        self.run_button.configure(state="normal")
        self.is_running = False

    def set_running_false(self):
        self.is_running = False
        self.run_button.configure(state="normal")

    def exit_app(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = BeeTrackingApp(root)
    root.mainloop()