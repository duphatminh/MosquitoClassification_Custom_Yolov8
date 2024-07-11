from multiprocessing import freeze_support
from ultralytics import YOLO

class Train_model_Yolov8:
    def train_model_det(self):
        model_det = YOLO("yolov8n.pt")
        results_det = model_det.train(data="data/data.yaml", epochs=50, device="cuda:0")

        return results_det

    def train_model_seg(self):
        model_seg = YOLO("yolov8n.pt")
        results_seg = model_seg.train(data="data/data.yaml", epochs=50, device="cuda:0")

        return results_seg

if __name__ == '__main__':
    freeze_support()
    train = Train_model_Yolov8()

    train.train_model_det()
    #train.train_model_seg()
