import argparse
import time
from pathlib import Path

import cv2
import yaml

from src.model_manager import ensure_model
from src.detector import PPEModel
from src.compliance import extract, is_person, analyze_person
from src.logger import ViolationLogger
from src.utils import draw, snapshot, banner


def load_settings():
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def source_value(value):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser(description="PPE / Helmet Detection System V2")
    parser.add_argument("--source", default="0")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--save-output", action="store_true")
    args = parser.parse_args()

    cfg = load_settings()
    mcfg = cfg["model"]
    icfg = cfg["inference"]
    rcfg = cfg["rules"]
    ocfg = cfg["output"]

    model_path = ensure_model(
        mcfg["repo_id"],
        mcfg["filename"],
        mcfg["local_path"],
    )

    confidence = args.conf if args.conf is not None else icfg["confidence"]
    model = PPEModel(model_path, confidence, icfg["iou"], icfg["image_size"])
    logger = ViolationLogger(ocfg["log_csv"])

    source = source_value(args.source)

    # Image mode
    if isinstance(source, str) and Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }:
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"Could not read image: {source}")

        result = model.predict(frame)
        detections = extract(result)
        persons = [d for d in detections if is_person(d.name)]

        violation = False
        violation_types = []
        for d in detections:
            draw(frame, d)

        for person in persons:
            status = analyze_person(
                person,
                detections,
                rcfg["association_margin"],
                rcfg["missing_vest_enabled"],
            )
            if status["violation"]:
                violation = True
                if status["helmet_violation"]:
                    violation_types.append("NO_HELMET")
                if status["vest_violation"]:
                    violation_types.append("NO_VEST")

        out = Path(ocfg["annotated_dir"])
        out.mkdir(parents=True, exist_ok=True)
        output_path = out / f"annotated_{Path(source).stem}.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    writer = None
    last_violation = 0.0
    total_violations = 0
    compliant_people = 0
    fps = 0.0
    previous = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame)
        detections = extract(result)
        persons = [
            d for d in detections
            if is_person(d.name) and d.confidence >= rcfg["minimum_person_confidence"]
        ]

        for d in detections:
            draw(frame, d)

        violation_found = False
        violation_types = []
        compliant_people = 0

        for person in persons:
            status = analyze_person(
                person,
                detections,
                rcfg["association_margin"],
                rcfg["missing_vest_enabled"],
            )
            if status["violation"]:
                violation_found = True
                if status["helmet_violation"]:
                    violation_types.append("NO_HELMET")
                if status["vest_violation"]:
                    violation_types.append("NO_VEST")
            else:
                compliant_people += 1

        now = time.time()
        if violation_found and now - last_violation >= rcfg["violation_cooldown_seconds"]:
            snap = snapshot(frame, ocfg["violation_dir"])
            violation_conf = max(
                [d.confidence for d in detections
                 if "no-helmet" in d.name.lower() or "no_helmet" in d.name.lower()]
                or [0.0]
            )

            # If the violation is a missing vest inferred from absence, use person confidence.
            if "NO_VEST" in violation_types:
                violation_conf = max(
                    violation_conf,
                    max([p.confidence for p in persons] or [0.0])
                )

            for v in sorted(set(violation_types)):
                logger.add(v, violation_conf, str(source), snap)

            total_violations += 1
            last_violation = now

        current = time.time()
        delta = current - previous
        if delta > 0:
            instant = 1.0 / delta
            fps = instant if fps == 0 else fps * 0.9 + instant * 0.1
        previous = current

        banner(frame, f"FPS: {fps:.1f} | Workers: {len(persons)}", 30, (255, 255, 0))
        banner(
            frame,
            "VIOLATION DETECTED" if violation_found else "PPE COMPLIANT",
            60,
            (0, 0, 255) if violation_found else (0, 200, 0),
        )
        banner(
            frame,
            f"Compliant: {compliant_people} | Incidents: {total_violations}",
            90,
            (255, 255, 255),
        )

        if args.save_output:
            if writer is None:
                h, w = frame.shape[:2]
                out_dir = Path(ocfg["annotated_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(out_dir / "detection_output.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20.0,
                    (w, h),
                )
            writer.write(frame)

        cv2.imshow("PPE / Helmet Detection System V2", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
