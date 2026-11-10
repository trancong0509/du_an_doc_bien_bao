from pathlib import Path

from ultralytics import YOLO


def main():
    project_dir = Path(__file__).resolve().parent
    model = YOLO("yolo11n.pt")

    model.train(
        data=str(project_dir / "data.yaml"),
        epochs=1,
        imgsz=320,
        batch=2,
        device="cpu",
        workers=0,
        fliplr=0.0,
        project=str(project_dir / "runs"),
        name="bien_bao",
    )

    print("Mô hình tốt nhất:", model.trainer.best)


if __name__ == "__main__":
    main()