from __future__ import annotations

import math
import time

import cv2
import numpy as np


class DemoFloodSource:
    def __init__(self, width: int = 960, height: int = 540) -> None:
        self.width = width
        self.height = height
        self.opened = True

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (48, 55, 62)

        # 도로
        road_top = int(self.height * 0.34)
        pts = np.array([
            [int(self.width * 0.34), road_top],
            [int(self.width * 0.66), road_top],
            [int(self.width * 0.95), self.height],
            [int(self.width * 0.05), self.height],
        ])
        cv2.fillPoly(frame, [pts], (65, 68, 72))

        # 차선
        for offset in (-90, 90):
            cv2.line(
                frame,
                (self.width // 2 + offset // 3, road_top),
                (self.width // 2 + offset, self.height),
                (210, 210, 210),
                3,
            )

        # 차량
        x1, y1 = int(self.width * 0.36), int(self.height * 0.36)
        x2, y2 = int(self.width * 0.64), int(self.height * 0.84)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 80, 175), -1)
        cv2.rectangle(frame, (x1 + 30, y1 + 25), (x2 - 30, y1 + 105), (100, 145, 175), -1)
        cv2.circle(frame, (x1 + 35, y2), 31, (20, 20, 20), -1)
        cv2.circle(frame, (x2 - 35, y2), 31, (20, 20, 20), -1)

        # 시간에 따라 변하는 수면
        t = time.monotonic()
        water_ratio = (math.sin(t / 5.0) + 1) / 2
        water_y = int(self.height - water_ratio * self.height * 0.48)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, water_y), (self.width, self.height), (175, 105, 45), -1)
        frame = cv2.addWeighted(overlay, 0.50, frame, 0.50, 0)

        for i in range(8):
            y = water_y + i * 24
            dx = int(20 * math.sin(t * 2 + i))
            cv2.line(frame, (80 + dx, y), (self.width - 80 + dx, y), (200, 155, 105), 2)

        cv2.putText(
            frame, "DEMO CCTV / SYNTHETIC FLOOD",
            (20, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA
        )
        return True, frame

    def release(self) -> None:
        self.opened = False
