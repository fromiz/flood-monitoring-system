from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.anonymizer import safe_status


def main() -> None:
    result = safe_status()
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    if not result.get("operational"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
