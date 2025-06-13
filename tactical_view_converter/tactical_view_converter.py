import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .homography import Homography

folder = Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder, "../"))
from utils import get_foot_position, measure_distance  # type: ignore

BBox = Tuple[int, int, int, int]
TrackFrame = Dict[int, Dict[str, BBox]]
KeypointsFrame = Any  # whatever Ultralytics keypoints structure is


class TacticalViewConverter:
    def __init__(self, courtImagePath: str | Path) -> None:
        self.courtImagePath = str(courtImagePath)
        self.width, self.height = 300, 161
        self.actualWidthM, self.actualHeightM = 28.0, 15.0
        h, w = self.height, self.width
        ah, aw = self.actualHeightM, self.actualWidthM
        self.keyPoints: List[Tuple[int, int]] = [
            (0, 0),
            (0, int(0.91 / ah * h)),
            (0, int(5.18 / ah * h)),
            (0, int(10 / ah * h)),
            (0, int(14.1 / ah * h)),
            (0, h),
            (w // 2, h),
            (w // 2, 0),
            (int(5.79 / aw * w), int(5.18 / ah * h)),
            (int(5.79 / aw * w), int(10 / ah * h)),
            (w, h),
            (w, int(14.1 / ah * h)),
            (w, int(10 / ah * h)),
            (w, int(5.18 / ah * h)),
            (w, int(0.91 / ah * h)),
            (w, 0),
            (int((aw - 5.79) / aw * w), int(5.18 / ah * h)),
            (int((aw - 5.79) / aw * w), int(10 / ah * h)),
        ]

    def validateKeypoints(self, frames: Sequence[KeypointsFrame]):
        kpList = deepcopy(frames)
        for idx, fkp in enumerate(kpList):
            kps = fkp.xy.tolist()[0]
            detected = [i for i, p in enumerate(kps) if p[0] > 0 and p[1] > 0]
            if len(detected) < 3:
                continue
            invalid: List[int] = []
            for i in detected:
                others = [j for j in detected if j != i and j not in invalid]
                if len(others) < 2:
                    continue
                j, k = others[:2]
                dij = measure_distance(kps[i], kps[j])
                dik = measure_distance(kps[i], kps[k])
                tij = measure_distance(self.keyPoints[i], self.keyPoints[j])
                tik = measure_distance(self.keyPoints[i], self.keyPoints[k])
                if tik == 0 or dik == 0 or tij == 0:
                    continue
                err = abs(dij / dik - tij / tik)
                if err > 0.8:
                    fkp.xy[0][i] *= 0
                    fkp.xyn[0][i] *= 0
                    invalid.append(i)
        return kpList

    def transformPlayers(
        self,
        keypoints: Sequence[KeypointsFrame],
        tracks: Sequence[TrackFrame],
    ) -> List[Dict[int, List[float]]]:
        out: List[Dict[int, List[float]]] = []
        for fkp, tr in zip(keypoints, tracks):
            tact: Dict[int, List[float]] = {}
            kps = fkp.xy.tolist()[0]
            valid = [i for i, p in enumerate(kps) if p[0] > 0 and p[1] > 0]
            if len(valid) < 4:
                out.append(tact)
                continue
            src = np.array([kps[i] for i in valid], dtype=np.float32)
            tgt = np.array([self.keyPoints[i] for i in valid], dtype=np.float32)
            try:
                H = Homography(src, tgt)
                for pid, data in tr.items():
                    foot = np.array([get_foot_position(data["bbox"])], dtype=np.float32)
                    tp = H.transformPoints(foot)[0]
                    if 0 <= tp[0] <= self.width and 0 <= tp[1] <= self.height:
                        tact[pid] = tp.tolist()
            except Exception:
                pass
            out.append(tact)
        return out
