from typing import Any, Dict, List, Sequence

import supervision as sv
from ultralytics import YOLO
import sys

sys.path.append("../")
from utils import read_stub, save_stub  # type: ignore

Frame = Any
TrackFrame = Dict[int, Dict[str, List[float]]]


class PlayerTracker:
    def __init__(self, modelPath: str) -> None:
        self.model = YOLO(modelPath)
        self.tracker = sv.ByteTrack()

    def _detectBatch(self, frames: Sequence[Frame], conf: float = 0.5, batch: int = 20):
        detected = []
        for i in range(0, len(frames), batch):
            detected.extend(self.model.predict(frames[i : i + batch], conf=conf))
        return detected

    def objectTracks(
        self,
        frames: Sequence[Frame],
        *,
        readFromStub: bool = False,
        stubPath: str | None = None,
    ) -> List[TrackFrame]:
        cached = read_stub(readFromStub, stubPath)
        if cached is not None and len(cached) == len(frames):
            return cached  # type: ignore[return-value]
        detections = self._detectBatch(frames)
        out: List[TrackFrame] = []
        for det in detections:
            names = {v: k for k, v in det.names.items()}
            supDet = sv.Detections.from_ultralytics(det)
            trks = self.tracker.update_with_detections(supDet)
            frameDict: TrackFrame = {}
            for d in trks:
                bbox, clsId, trkId = d[0].tolist(), d[3], d[4]
                if clsId == names["Player"]:
                    frameDict[trkId] = {"bbox": bbox}
            out.append(frameDict)
        save_stub(stubPath, out)
        return out
