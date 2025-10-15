from ultralytics import YOLO
import yaml
import os

# -----------------------------
# 1. Data setup
# -----------------------------
# Your dataset should be structured as:
# dataset/
#   images/
#     train/
#     val/
#   labels/
#     train/
#     val/
# Each label is in YOLO format: class x_center y_center width height (normalized)

data_yaml = {
    'path': 'scripts/new_datasetV2',  # root dir
    'train': 'train/images',
    'val': 'valid/images',
    'nc': 1,             # number of classes (1 for bees)
    'names': ['honeybee']     # class name
}

# Save data.yaml
#with open('data_bees.yaml', 'w') as f:
#    yaml.dump(data_yaml, f)

# -----------------------------
# 2. Training hyperparameters
# -----------------------------
# These are good defaults for small objects like bees
training_params = {
    'data': 'scripts/new_datasetV2/data.yaml',
    'model': 'yolov8m.pt',  # small pretrained model, more capacity than n
    'epochs': 200,           # usually 150-200 enough with good data
    'imgsz': 640,           # high-res helps tiny objects
    'batch': 8,              # adjust to GPU memory
    'lr0': 1e-3,             # initial learning rate
        'augment': True,         # enable augmentation
        'mosaic': True,           # mosaic augmentation
    'mixup': 0.2,             # optional: mixup helps
    'save_period': 10,        # save checkpoints every 10 epochs
    'patience': 50,            # early stopping patience
    'workers': 0
}

# -----------------------------
# 3. Initialize and train
# -----------------------------
def train_bee_detector():
    model = YOLO(training_params['model'])  # load pretrained
    results = model.train(
        data=training_params['data'],
        epochs=training_params['epochs'],
        imgsz=training_params['imgsz'],
        batch=training_params['batch'],
        lr0=training_params['lr0'],
        augment=training_params['augment'],
        mosaic=training_params['mosaic'],
        mixup=training_params['mixup'],
        save_period=training_params['save_period'],
        patience=training_params['patience'],
        workers=training_params['workers']
    )
    print("Training finished.")
    return model

# -----------------------------
# 4. Inference with tuned confidence
# -----------------------------
def run_inference(model_path, image_path):
    model = YOLO(model_path)
    # Lower confidence threshold for tiny objects
    results = model.predict(
        source=image_path,
        conf=0.25,        # lower threshold, catches more bees
        iou=0.5,
        imgsz=1024
    )
    # Visualize predictions
    results.show()

if __name__ == "__main__":
    model = train_bee_detector()
    run_inference("runs/train/exp/weights/best.pt", "dataset/images/val/sample.jpg")
