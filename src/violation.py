from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Detection:
    cls_name: str
    confidence: float
    box: tuple  # x1, y1, x2, y2


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def inside_or_near(inner, outer, margin=0.15):
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    ow = ox2 - ox1
    oh = oy2 - oy1

    return (
        ix1 >= ox1 - ow * margin and
        iy1 >= oy1 - oh * margin and
        ix2 <= ox2 + ow * margin and
        iy2 <= oy2 + oh * margin
    )


def analyze_person(person: Detection, detections: List[Detection], margin=0.15) -> Dict:
    nearby = [
        d for d in detections
        if d is not person and inside_or_near(d.box, person.box, margin)
    ]

    helmet = any(d.cls_name == "helmet" for d in nearby)
    no_helmet = any(d.cls_name == "no_helmet" for d in nearby)
    vest = any(d.cls_name == "safety_vest" for d in nearby)
    no_vest = any(d.cls_name == "no_vest" for d in nearby)

    return {
        "helmet": helmet,
        "no_helmet": no_helmet,
        "vest": vest,
        "no_vest": no_vest,
        "helmet_violation": no_helmet and not helmet,
        "vest_violation": no_vest and not vest,
        "violation": (no_helmet and not helmet) or (no_vest and not vest),
    }


def extract_detections(result) -> List[Detection]:
    detections = []
    if result.boxes is None:
        return detections

    names = result.names
    for box, conf, cls_id in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist()
    ):
        cls_id = int(cls_id)
        detections.append(
            Detection(
                cls_name=str(names[cls_id]),
                confidence=float(conf),
                box=tuple(map(int, box))
            )
        )
    return detections
