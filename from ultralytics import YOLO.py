from ultralytics import YOLO

model = YOLO("nut_bolt_model.pt")
print(model.model.args)  # モデルの内部構成を見る