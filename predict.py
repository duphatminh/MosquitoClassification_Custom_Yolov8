from ultralytics import YOLO
from PIL import Image

model = YOLO("runs/segment/train/weights/best.pt")
results = model("data/test/images/How-Mosquitoes-Get-Away_100_jpg.rf.1a3f8a0c3c4fd37784196d0a4f31df40.jpg")

for i in results:
    print(i.boxes)
    im_array = i.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save("outputs/output.jpg")