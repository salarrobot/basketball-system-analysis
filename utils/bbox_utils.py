from typing import Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def centerOfBbox(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def bboxWidth(bbox: BBox) -> float:
    x1, _, x2, _ = bbox
    return x2 - x1


def distance(p1: Point, p2: Point) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def xyDistance(p1: Point, p2: Point) -> Point:
    return p1[0] - p2[0], p1[1] - p2[1]


def footPosition(bbox: BBox) -> Point:
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2, y2