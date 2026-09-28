from pathlib import Path
import time
import cv2


def color_for(name):
    n = str(name).lower()
    if "no-" in n or "no_" in n:
        return (0, 0, 255)
    if n in {"human", "person", "worker"}:
        return (255, 190, 0)
    return (0, 210, 0)


def draw(frame, detection, violation=False):
    x1, y1, x2, y2 = detection.box
    color = (0, 0, 255) if violation else color_for(detection.name)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{detection.name} {detection.confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(22, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def snapshot(frame, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"violation_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def banner(frame, text, y, color):
    cv2.putText(
        frame,
        text,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )
