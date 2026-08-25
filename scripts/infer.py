from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.inference import FloodDetector, draw_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="outputs/result.mp4")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"입력을 열 수 없습니다: {args.source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    detector = FloodDetector(args.weights, confidence=args.conf, device=args.device)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = detector.predict(frame)
            writer.write(draw_result(frame, result))
    finally:
        capture.release()
        writer.release()

    print(f"저장 완료: {output}")


if __name__ == "__main__":
    main()
