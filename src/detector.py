from pathlib import Path
from ultralytics import YOLO


class PPEModel:
    def __init__(self, model_path: str, confidence: float, iou: float, image_size: int = 640):
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size

    @property
    def names(self):
        return self.model.names

    def predict(self, frame):
        return self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            verbose=False,
        )[0]
