from multiprocessing import freeze_support
from ultralytics import YOLO


class Train_model_Yolov8:
    def __init__(self, data_path="data/data.yaml", epochs=50, device="cuda:0", batch=16, lr0=0.001):
        self.data_path = data_path
        self.epochs = epochs
        self.device = device
        self.batch = batch
        self.lr0 = lr0

    def train_model_det(self):
        model_det = YOLO("yolov8n.pt")
        results_det = model_det.train(
            data=self.data_path,
            epochs=self.epochs,
            batch=self.batch,
            lr0=self.lr0,
            device=self.device
        )
        return results_det

    def train_model_seg(self):
        model_seg = YOLO("yolov8n-seg.pt")

        results_seg = model_seg.train(
            data=self.data_path,
            epochs=self.epochs,
            batch=self.batch,
            lr0=self.lr0,
            device=self.device
        )
        return results_seg


if __name__ == '__main__':
    freeze_support()
    train = Train_model_Yolov8(data_path="data/data.yaml", epochs=50, batch=40, lr0=0.001)

    results_det = train.train_model_det()
    results_seg = train.train_model_seg()
