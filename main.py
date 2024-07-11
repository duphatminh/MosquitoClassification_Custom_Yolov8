if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    from ultralytics import YOLO

    model_det = YOLO("yolov8n.pt")
    results_det = model_det.train(data="data/data.yaml", epochs=50, device="cuda:0")
    
    # model_seg = YOLO("yolov8n-seg.pt")
    # results_seg = model_seg.train(data="data/data.yaml", epochs=50, device="cuda:0")