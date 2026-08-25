from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="YOLO 침수 데이터셋 검사")
    parser.add_argument("--root", default="datasets/flood")
    args = parser.parse_args()

    root = Path(args.root)
    errors: list[str] = []
    counts = Counter()

    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.exists():
            errors.append(f"누락: {image_dir}")
            continue

        images = [p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"라벨 없음: {image}")
                continue
            for line_no, line in enumerate(label.read_text().splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label}:{line_no} 필드 개수 오류")
                    continue
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                if class_id not in range(5):
                    errors.append(f"{label}:{line_no} 클래스 오류 {class_id}")
                if any(v < 0 or v > 1 for v in coords):
                    errors.append(f"{label}:{line_no} 좌표 범위 오류")
                counts[(split, class_id)] += 1

    print("클래스 분포")
    for key in sorted(counts):
        print(key, counts[key])
    if errors:
        print(f"\n오류 {len(errors)}건")
        for error in errors[:100]:
            print("-", error)
        raise SystemExit(1)
    print("\n데이터셋 검사 통과")


if __name__ == "__main__":
    main()
