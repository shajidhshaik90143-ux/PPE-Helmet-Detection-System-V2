from dataclasses import dataclass


@dataclass
class Detection:
    name: str
    confidence: float
    box: tuple


def normalize(name: str) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def is_person(name: str) -> bool:
    return normalize(name) in {"human", "person", "worker"}


def is_helmet(name: str) -> bool:
    return normalize(name) in {"helmet", "hardhat", "hard-hat"}


def is_no_helmet(name: str) -> bool:
    return normalize(name) in {"no-helmet", "no-hardhat", "no-hard-hat"}


def is_vest(name: str) -> bool:
    return normalize(name) in {"vest", "safety-vest", "safetyvest"}


def overlap_ratio(inner, outer):
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer

    ix1, iy1 = max(ix1, ox1), max(iy1, oy1)
    ix2, iy2 = min(ix2, ox2), min(iy2, oy2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / inner_area


def near(person_box, object_box, margin=0.20):
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = object_box

    pw = px2 - px1
    ph = py2 - py1

    expanded = (
        px1 - pw * margin,
        py1 - ph * margin,
        px2 + pw * margin,
        py2 + ph * margin,
    )

    ex1, ey1, ex2, ey2 = expanded
    return ox1 >= ex1 and oy1 >= ey1 and ox2 <= ex2 and oy2 <= ey2


def analyze_person(person: Detection, detections, margin=0.20, missing_vest=True):
    related = [
        d for d in detections
        if d is not person and near(person.box, d.box, margin)
    ]

    helmet = any(is_helmet(d.name) for d in related)
    no_helmet = any(is_no_helmet(d.name) for d in related)
    vest = any(is_vest(d.name) for d in related)

    helmet_violation = no_helmet and not helmet
    vest_violation = missing_vest and not vest

    return {
        "helmet": helmet,
        "no_helmet": no_helmet,
        "vest": vest,
        "helmet_violation": helmet_violation,
        "vest_violation": vest_violation,
        "violation": helmet_violation or vest_violation,
    }


def extract(result):
    detections = []
    if result.boxes is None:
        return detections

    names = result.names
    for box, conf, cls_id in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        cls_id = int(cls_id)
        detections.append(
            Detection(
                name=str(names[cls_id]),
                confidence=float(conf),
                box=tuple(int(v) for v in box),
            )
        )
    return detections
