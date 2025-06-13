from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from ultralytics import YOLO
import supervision as sv  # retained import for side‑effects/typing

sys.path.append("../")
from utils import read_stub, save_stub

Frame = np.ndarray
CourtKeypoints = Any


class CourtKeypointDetector:
    def __init__(self, modelPath: str | Path) -> None:
        self.model = YOLO(str(modelPath))

    def getCourtKeypoints(
        self,
        frames: Sequence[Frame],
        *,
        readFromStub: bool = False,
        stubPath: Optional[str | Path] = None,
        batchSize: int = 20,
        conf: float = 0.5,
    ) -> List[CourtKeypoints]:
        maybeCached = read_stub(readFromStub, stubPath)
        if maybeCached is not None and len(maybeCached) == len(frames):
            return maybeCached

        courtKeypoints: List[CourtKeypoints] = []
        for idx in range(0, len(frames), batchSize):
            batch = frames[idx : idx + batchSize]
            detections = self.model.predict(list(batch), conf=conf)
            for det in detections:
                courtKeypoints.append(det.keypoints)

        save_stub(stubPath, courtKeypoints)
        return courtKeypoints
