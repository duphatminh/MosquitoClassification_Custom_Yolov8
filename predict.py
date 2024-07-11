from ultralytics import YOLO
from PIL import Image

model_det = YOLO("runs/detect/train/weights/best.pt")
results_det = model_det("data/test/images/How-Mosquitoes-Get-Away_100_jpg.rf.1a3f8a0c3c4fd37784196d0a4f31df40.jpg")

model_seg = YOLO("runs/segment/train/weights/best.pt")
results_seg = model_seg("data/test/images/How-Mosquitoes-Get-Away_100_jpg.rf.1a3f8a0c3c4fd37784196d0a4f31df40.jpg")


for i in results_det:
    print(i.boxes)
    im_array = i.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save("outputs/output_det.jpg")

for i in results_seg:
    print(i.boxes)
    im_array = i.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save("outputs/output_seg.jpg")