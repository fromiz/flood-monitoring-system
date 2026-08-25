"""
실시간 얼굴+번호판 모자이크 (C 모델 = dashcam_anonymizer 단일 YOLOv8)
========================================================================
웹캠 / RTSP·HTTP 스트림 / 영상파일을 실시간으로 받아
탐지된 얼굴·번호판을 즉시 모자이크해 화면에 띄웁니다.

실행:
    python realtime_anonymizer.py                      # 기본 웹캠(0)
    python realtime_anonymizer.py --source 1           # 두 번째 카메라
    python realtime_anonymizer.py --source rtsp://...  # IP 카메라
    python realtime_anonymizer.py --source clip.mp4    # 영상 파일
    python realtime_anonymizer.py --record out.mp4     # 결과 녹화까지

창에서:  q=종료   s=스냅샷 저장   r=녹화 토글   스페이스=일시정지

사전 준비 (모델 1개만 있으면 됩니다):
    pip install ultralytics opencv-python
    # dashcam 모델(best.pt) 다운로드 → --model 경로로 지정
    #   gdown:  pip install gdown && gdown 1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe -O best.pt
    #   또는 레포의 SharePoint 링크에서 수동 다운로드:
    #   https://github.com/varungupta31/dashcam_anonymizer
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 모자이크 함수
# ---------------------------------------------------------------------------
def pixelate(img, x1, y1, x2, y2, blocks=12):
    """영역을 블록 단위로 픽셀레이션. 블러보다 복원 저항성이 높아 비식별화에 적합."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return img
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(
        roi,
        (max(1, (x2 - x1) // blocks), max(1, (y2 - y1) // blocks)),
        interpolation=cv2.INTER_LINEAR,
    )
    img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return img


def gaussian(img, x1, y1, x2, y2, _blocks=None):
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return img
    k = max(3, (min(x2 - x1, y2 - y1) // 3) | 1)  # 홀수 커널
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)
    return img


def expand(box, scale, shape):
    """박스를 살짝 키워 경계 누출(가장자리 글자/얼굴 일부 노출) 방지."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    h, w = shape[:2]
    return (
        max(0, cx - bw / 2), max(0, cy - bh / 2),
        min(w, cx + bw / 2), min(h, cy + bh / 2),
    )


# ---------------------------------------------------------------------------
# 소스 열기 (웹캠 인덱스 / 스트림 URL / 파일 경로 자동 판별)
# ---------------------------------------------------------------------------
def open_source(src: str):
    if src.isdigit():                       # "0", "1" → 웹캠 인덱스
        idx = int(src)
        # Windows는 CAP_DSHOW가 지연이 적음
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(idx, backend)
    else:                                   # URL 또는 파일
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"소스를 열 수 없습니다: {src} (카메라 연결/URL/경로 확인)")
    return cap


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="실시간 얼굴+번호판 모자이크 (dashcam C 모델)")
    ap.add_argument("--source", default="0", help='웹캠 인덱스("0"), RTSP/HTTP URL, 또는 영상 파일')
    ap.add_argument("--model", default="best.pt", help="dashcam YOLOv8 가중치 경로")
    ap.add_argument("--device", default="0", help='GPU면 "0", CPU면 "cpu"')
    ap.add_argument("--conf", type=float, default=0.10, help="신뢰도 임계값(낮을수록 더 많이 가림)")
    ap.add_argument("--imgsz", type=int, default=640, help="추론 해상도(작을수록 빠름)")
    ap.add_argument("--method", default="pixelate", choices=["pixelate", "gaussian"])
    ap.add_argument("--blocks", type=int, default=12, help="픽셀레이션 강도(작을수록 굵게)")
    ap.add_argument("--scale", type=float, default=1.20, help="박스 확대 배율")
    ap.add_argument("--record", default=None, help="결과를 저장할 mp4 경로(옵션)")
    ap.add_argument("--max-width", type=int, default=1280, help="표시/처리 최대 가로폭(초과 시 축소)")
    ap.add_argument("--skip", type=int, default=0, help="N프레임마다 1번만 추론(끊기면 1~2로)")
    ap.add_argument("--no-display", action="store_true", help="창 없이 처리(헤드리스 서버용)")
    args = ap.parse_args()

    # --- 모델 로드 ---
    if not Path(args.model).exists():
        sys.exit(
            f"[ERROR] 모델 파일이 없습니다: {args.model}\n"
            "  dashcam best.pt 를 받아 --model 로 지정하세요:\n"
            "    pip install gdown\n"
            "    gdown 1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe -O best.pt\n"
            "  (실패 시 https://github.com/varungupta31/dashcam_anonymizer 의 SharePoint 링크에서 수동 다운로드)"
        )
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics 미설치 → pip install ultralytics opencv-python")

    print(f"[load] {args.model}")
    model = YOLO(args.model)
    print(f"[info] 클래스: {model.names}  (탐지된 건 모두 모자이크 처리)")

    apply_mosaic = pixelate if args.method == "pixelate" else gaussian

    # --- 소스 열기 ---
    cap = open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    recording = False

    def make_writer(w, h, path):
        return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

    if args.record:                         # 시작부터 녹화
        recording = True

    fps_hist = deque(maxlen=30)
    frame_id = 0
    last_boxes = []                         # skip 프레임에서 재사용할 직전 박스
    paused = False
    win = "Realtime Anonymizer (q=quit  s=snap  r=rec  space=pause)"

    print("[run] 시작. 창에서 q로 종료.")
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("[end] 스트림 종료 / 프레임 없음")
                    break

                # 큰 프레임은 축소해 실시간성 확보
                if frame.shape[1] > args.max_width:
                    r = args.max_width / frame.shape[1]
                    frame = cv2.resize(frame, (args.max_width, int(frame.shape[0] * r)))

                t0 = time.time()

                # skip 옵션: 무거우면 N프레임마다만 추론하고 사이엔 직전 박스 재사용
                do_infer = (args.skip == 0) or (frame_id % (args.skip + 1) == 0)
                if do_infer:
                    res = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                        device=args.device, verbose=False)[0]
                    last_boxes = [b.tolist() for b in res.boxes.xyxy.cpu()]

                for b in last_boxes:
                    bx = expand(b, args.scale, frame.shape)
                    frame = apply_mosaic(frame, *bx, args.blocks)

                dt = time.time() - t0
                fps_hist.append(1.0 / dt if dt > 0 else 0.0)
                fps = sum(fps_hist) / len(fps_hist)

                # HUD
                cv2.putText(frame, f"FPS:{fps:5.1f}  det:{len(last_boxes)}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if recording:
                    cv2.circle(frame, (frame.shape[1] - 25, 25), 9, (0, 0, 255), -1)

                # 녹화
                if recording:
                    if writer is None:
                        path = args.record or f"rec_{int(time.time())}.mp4"
                        writer = make_writer(frame.shape[1], frame.shape[0], path)
                        print(f"[rec] 녹화 시작 → {path}")
                    writer.write(frame)

                frame_id += 1

            if not args.no_display:
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    snap = f"snap_{int(time.time())}.jpg"
                    cv2.imwrite(snap, frame)
                    print(f"[snap] 저장 → {snap}")
                elif key == ord("r"):
                    recording = not recording
                    if not recording and writer is not None:
                        writer.release(); writer = None
                        print("[rec] 녹화 중지")
                elif key == ord(" "):
                    paused = not paused
            else:
                # 헤드리스: 파일 소스가 끝나면 자동 종료, 스트림은 Ctrl+C
                pass
    except KeyboardInterrupt:
        print("\n[stop] 사용자 중단")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("[done] 정리 완료")


if __name__ == "__main__":
    main()
