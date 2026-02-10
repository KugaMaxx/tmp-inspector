from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11x.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="./data/qwen-image-fire-detector+extinguisher/data.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=1024,  # Image size for training
    device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()
