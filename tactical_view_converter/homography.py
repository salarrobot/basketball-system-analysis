import numpy as np
import cv2
from typing import Any


class Homography:
    def __init__(self, src: np.ndarray, dst: np.ndarray) -> None:
        if src.shape != dst.shape:
            raise ValueError("Source and target must have the same shape.")
        if src.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")
        src_f = src.astype(np.float32)
        dst_f = dst.astype(np.float32)
        m, _ = cv2.findHomography(src_f, dst_f)
        if m is None:
            raise ValueError("Homography matrix could not be calculated.")
        self.m = m

    def transformPoints(self, pts: np.ndarray) -> np.ndarray:  # noqa: N802
        if pts.size == 0:
            return pts
        if pts.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")
        pts_f = pts.reshape(-1, 1, 2).astype(np.float32)
        res = cv2.perspectiveTransform(pts_f, self.m)
        return res.reshape(-1, 2).astype(np.float32)
