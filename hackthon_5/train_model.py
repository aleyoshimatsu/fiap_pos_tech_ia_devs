import os
from ultralytics import YOLO

BASE = os.path.dirname(__file__)
DATA_YAML = os.path.join(BASE, 'data', 'cortantes.yaml')
WEIGHTS = os.path.join(BASE, 'yolov5su.pt')
model = YOLO(WEIGHTS)
model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    name='cortantes_experiment'
)
