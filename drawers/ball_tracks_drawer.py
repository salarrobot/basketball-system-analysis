from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .utils import draw_traingle

Frame = np.ndarray
TrackInfo = Dict[str, Any]
TracksFrame = Dict[Any, TrackInfo]


class BallTracksDrawer:
    def __init__(self, pointerColor: Tuple[int, int, int] = (0, 255, 0)) -> None:
        self.pointerColor = pointerColor

    def drawTracks(self, videoFrames: Sequence[Frame], tracks: Sequence[TracksFrame]) -> List[Frame]:
        output: List[Frame] = []
        for idx, frame in enumerate(videoFrames):
            f = frame.copy()
            for ball in tracks[idx].values():
                bbox = ball.get("bbox")
                if bbox is None:
                    continue
                f = draw_traingle(f, bbox, self.pointerColor)
            output.append(f)
        return output
