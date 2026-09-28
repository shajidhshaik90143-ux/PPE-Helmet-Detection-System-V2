from ultralytics import YOLO

# Ultralytics Construction-PPE dataset.
# This downloads the dataset automatically when training begins.
# It contains 11 classes including helmet, vest, Person, no_helmet, gloves,
# boots and goggles.

MODEL = "yolo26n.pt"
DATA = "construction-ppe.yaml"


def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA,
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,
        project="runs",
        name="construction_ppe",
    )


if __name__ == "__main__":
    main()
