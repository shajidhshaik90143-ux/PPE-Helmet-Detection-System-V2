import csv
from datetime import datetime
from pathlib import Path


class ViolationLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp",
                    "violation_type",
                    "confidence",
                    "source",
                    "snapshot",
                ])

    def add(self, violation_type, confidence, source, snapshot):
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                violation_type,
                f"{confidence:.3f}",
                source,
                snapshot,
            ])
