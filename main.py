from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n-seg.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="function.yaml", epochs=10, device="cuda:0")    

