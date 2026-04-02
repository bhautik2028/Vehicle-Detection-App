from roboflow import Roboflow
from ultralytics import YOLO
import ultralytics
from ultralytics import settings
import torch

def main():
    # -------------------------------
    # Roboflow dataset download
    # -------------------------------
    rf = Roboflow(api_key="j06bCtB2G4wbHBfZTknY")

    project = rf.workspace("vehical-detection-and-counts") \
                .project("vehical_detection-7c3sr")

    version = project.version(1)
    dataset = version.download("yolov8")

    ultralytics.checks()

    settings.update({"sync": False})

    # -------------------------------
    # Load YOLO model
    # -------------------------------
    model = YOLO("yolov8l.pt")

    # -------------------------------
    # Train
    # -------------------------------
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=250,
        imgsz=960,
        batch=4,
        lr0=0.0003,
        optimizer="AdamW",
        mosaic=1.0,
        mixup=0.2,
        workers=4,   # 👈 safer on Windows
        device=0
    )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  
    main()
