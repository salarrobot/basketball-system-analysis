from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

Frame = np.ndarray


class FrameNumberDrawer:
    def __init__(
        self,
        color: Tuple[int, int, int] = (0, 255, 0),
        position: Tuple[int, int] = (10, 30),
        fontScale: float = 1.0,
        thickness: int = 2,
    ) -> None:
        self.color = color
        self.position = position
        self.fontScale = fontScale
        self.thickness = thickness

    def drawNumbers(self, frames: Sequence[Frame]) -> List[Frame]:
        out: List[Frame] = []
        for idx, frame in enumerate(frames):
            f = frame.copy()
            cv2.putText(
                f,
                str(idx),
                self.position,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.fontScale,
                self.color,
                self.thickness,
            )
            out.append(f)
        return out
