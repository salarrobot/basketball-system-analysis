from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

Frame = np.ndarray
TracksFrame = Dict[int, Dict[str, Any]]
DistanceFrame = Dict[int, float]
SpeedFrame = Dict[int, float]


class SpeedAndDistanceDrawer:
    def __init__(
        self,
        fontScale: float = 0.5,
        fontColor: Tuple[int, int, int] = (0, 0, 0),
        fontThickness: int = 2,
    ) -> None:
        self.fontScale = fontScale
        self.fontColor = fontColor
        self.fontThickness = fontThickness

    def drawMetrics(
        self,
        videoFrames: Sequence[Frame],
        playerTracks: Sequence[TracksFrame],
        playerDistances: Sequence[DistanceFrame],
        playerSpeeds: Sequence[SpeedFrame],
    ) -> List[Frame]:
        out: List[Frame] = []
        totalDistances: Dict[int, float] = {}
        for frame, tracks, distances, speeds in zip(
            videoFrames, playerTracks, playerDistances, playerSpeeds
        ):
            f = frame.copy()
            for pid, d in distances.items():
                totalDistances[pid] = totalDistances.get(pid, 0.0) + d
            for pid, info in tracks.items():
                x1, y1, x2, y2 = info["bbox"]
                px, py = int((x1 + x2) / 2), int(y2) + 40
                spd = speeds.get(pid)
                dist = totalDistances.get(pid)
                if spd is not None:
                    cv2.putText(
                        f,
                        f"{spd:.2f} km/h",
                        (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.fontScale,
                        self.fontColor,
                        self.fontThickness,
                    )
                if dist is not None:
                    cv2.putText(
                        f,
                        f"{dist:.2f} m",
                        (px, py + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.fontScale,
                        self.fontColor,
                        self.fontThickness,
                    )
            out.append(f)
        return out