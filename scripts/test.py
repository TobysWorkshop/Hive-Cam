from ultralytics import YOLO
model = YOLO("weights/entrance.pt")
result = model("data/videos/videoB.mp4")
print(result[0].masks)