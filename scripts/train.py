from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="침수 레벨 YOLO 모델 학습")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolov10n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/flood")
    parser.add_argument("--name", default="train")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError(f"data.yaml을 찾을 수 없습니다: {args.data}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        degrees=8.0,
        scale=0.35,
        translate=0.08,
        hsv_h=0.02,
        hsv_s=0.45,
        hsv_v=0.35,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=10,
        patience=25,
    )


if __name__ == "__main__":
    main()
