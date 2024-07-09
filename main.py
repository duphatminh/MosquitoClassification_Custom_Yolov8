if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    from ultralytics import YOLO

    model = YOLO("yolov8n-seg.pt")
    results = model.train(data="data/data.yaml", epochs=10, device="cuda:0")