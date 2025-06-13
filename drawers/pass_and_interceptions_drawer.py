from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

Frame = np.ndarray


class PassInterceptionDrawer:
    def __init__(
        self,
        fontScale: float = 0.7,
        fontThickness: int = 2,
        color: Tuple[int, int, int] = (0, 0, 0),
        overlayAlpha: float = 0.8,
    ) -> None:
        self.fontScale = fontScale
        self.fontThickness = fontThickness
        self.color = color
        self.overlayAlpha = overlayAlpha

    def getStats(self, passes: Sequence[int], interceptions: Sequence[int]) -> Tuple[int, int, int, int]:
        t1p = t2p = t1i = t2i = 0
        for p, inter in zip(passes, interceptions):
            if p == 1:
                t1p += 1
            elif p == 2:
                t2p += 1
            if inter == 1:
                t1i += 1
            elif inter == 2:
                t2i += 1
        return t1p, t2p, t1i, t2i

    def drawFrames(
        self,
        videoFrames: Sequence[Frame],
        passes: Sequence[int],
        interceptions: Sequence[int],
    ) -> List[Frame]:
        out: List[Frame] = []
        for idx, frame in enumerate(videoFrames):
            if idx == 0:
                out.append(frame.copy())
                continue
            f = frame.copy()
            self._drawFrame(f, idx, passes, interceptions)
            out.append(f)
        return out

    def _drawFrame(self, frame: Frame, idx: int, passes: Sequence[int], interceptions: Sequence[int]) -> None:
        h, w = frame.shape[:2]
        rx1, ry1 = int(w * 0.16), int(h * 0.75)
        rx2, ry2 = int(w * 0.55), int(h * 0.90)
        tx = int(w * 0.19)
        ty1, ty2 = int(h * 0.80), int(h * 0.88)
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, self.overlayAlpha, frame, 1 - self.overlayAlpha, 0, frame)
        p_until = passes[: idx + 1]
        i_until = interceptions[: idx + 1]
        t1p, t2p, t1i, t2i = self.getStats(p_until, i_until)
        cv2.putText(
            frame,
            f"Team 1 - Passes: {t1p} Interceptions: {t1i}",
            (tx, ty1),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.fontScale,
            self.color,
            self.fontThickness,
        )
        cv2.putText(
            frame,
            f"Team 2 - Passes: {t2p} Interceptions: {t2i}",
            (tx, ty2),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.fontScale,
            self.color,
            self.fontThickness,
        )
