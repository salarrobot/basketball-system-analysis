from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

sys.path.append("../")  # keep relative import workable when run as a script
from utils.bbox_utils import get_center_of_bbox, measure_distance  # noqa: E402

# ────────────────────────────────────────────────────────────────
# Type aliases
# ────────────────────────────────────────────────────────────────
BoundingBox = Tuple[int, int, int, int]  #: (x1, y1, x2, y2)
Point = Tuple[int, int]                   #: (x, y)
PlayerTrack = Dict[str, Any]              #: at least contains key ``"bbox"``
FrameTracks = Dict[int, PlayerTrack]      #: maps *player_id* → *PlayerTrack*
TracksOverTime = List[FrameTracks]        #: one entry per frame


class BallAquisitionDetector:
    """Detect which basketball player possesses the ball, frame‑by‑frame.

    The detector combines two cues:

    1. *Containment ratio* – the fraction of the ball's area overlapping
       a player's bounding box.
    2. *Proximity* – minimal Euclidean distance between the ball centre and
       carefully chosen key points on a player's box.

    A player must satisfy either a high containment ratio or be the closest
    within a configurable distance threshold for **``minFrames``** consecutive
    frames to be awarded possession.
    """

    # ------------------------------------------------------------------
    # Initialisation & configuration
    # ------------------------------------------------------------------
    def __init__(self,
                 possessionThreshold: int = 50,
                 minFrames: int = 11,
                 containmentThreshold: float = 0.8) -> None:
        self.possessionThreshold: int = possessionThreshold
        self.minFrames: int = minFrames
        self.containmentThreshold: float = containmentThreshold

    # ------------------------------------------------------------------
    # Geometry helpers (private)
    # ------------------------------------------------------------------
    def _getKeyPoints(self, playerBox: BoundingBox, ballCenter: Point) -> List[Point]:
        """Return key points around *playerBox*.

        The selection depends on where the ball lies relative to the box so that
        distance measures stay meaningful.
        """
        ballCx, ballCy = ballCenter
        x1, y1, x2, y2 = playerBox
        w, h = x2 - x1, y2 - y1

        pts: List[Point] = []

        # Points colinear with the ball centre if projection lands inside box.
        if y1 < ballCy < y2:
            pts.extend([(x1, ballCy), (x2, ballCy)])
        if x1 < ballCx < x2:
            pts.extend([(ballCx, y1), (ballCx, y2)])

        # Corner, edge‑midpoints & centre samplings.
        pts.extend([
            (x1 + w // 2, y1),            # top‑centre
            (x2, y1), (x1, y1),           # top‑right / top‑left
            (x2, y1 + h // 2),            # mid‑right
            (x1, y1 + h // 2),            # mid‑left
            (x1 + w // 2, y1 + h // 2),   # centre
            (x2, y2), (x1, y2),           # bottom‑right / bottom‑left
            (x1 + w // 2, y2),            # bottom‑centre
            (x1 + w // 2, y1 + h // 3),   # one‑third from top, centre x
        ])
        return pts

    @staticmethod
    def _containmentRatio(playerBox: BoundingBox, ballBox: BoundingBox) -> float:
        """Fraction of *ballBox* area lying inside *playerBox* (∈ [0, 1])."""
        px1, py1, px2, py2 = playerBox
        bx1, by1, bx2, by2 = ballBox

        ix1, iy1 = max(px1, bx1), max(py1, by1)
        ix2, iy2 = min(px2, bx2), min(py2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        ballArea = (bx2 - bx1) * (by2 - by1)
        return intersection / ballArea if ballArea else 0.0

    # ------------------------------------------------------------------
    # Frame‑level evaluation helpers
    # ------------------------------------------------------------------
    def _minDistance(self, ballCenter: Point, playerBox: BoundingBox) -> float:
        """Return minimal distance from *ballCenter* to player's key points."""
        return min(measure_distance(ballCenter, p) for p in self._getKeyPoints(playerBox, ballCenter))

    def _bestCandidate(self,
                       ballCenter: Point,
                       playerTracksFrame: FrameTracks,
                       ballBox: BoundingBox) -> int:
        """Player ID most likely to possess the ball in current frame, else ‑1."""
        containmentList: List[Tuple[int, float]] = []  # (id, dist)
        proximityList: List[Tuple[int, float]] = []

        for pid, info in playerTracksFrame.items():
            playerBox: BoundingBox | None = info.get("bbox")  # type: ignore[assignment]
            if not playerBox:
                continue

            ratio = self._containmentRatio(playerBox, ballBox)
            dist = self._minDistance(ballCenter, playerBox)

            (containmentList if ratio > self.containmentThreshold else proximityList).append((pid, dist))

        # Prefer high‑containment candidates – choose nearest among them.
        if containmentList:
            return min(containmentList, key=lambda t: t[1])[0]

        # Otherwise choose nearest within distance threshold.
        if proximityList:
            pid, d = min(proximityList, key=lambda t: t[1])
            if d < self.possessionThreshold:
                return pid
        return -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detectBallPossession(self,
                             playerTracks: TracksOverTime,
                             ballTracks: TracksOverTime) -> List[int]:
        """Return list *(possessionList)[frame] = playerId | ‑1*.

        The logic enforces that a player must satisfy the possession rules for
        *``minFrames``* consecutive frames before being recorded.
        """
        nFrames = len(ballTracks)
        possessionList: List[int] = [-1] * nFrames
        consecutive: Dict[int, int] = {}

        for f in range(nFrames):
            ballInfo: PlayerTrack | None = ballTracks[f].get(1)  # ball id assumed 1
            if not ballInfo or "bbox" not in ballInfo:
                consecutive.clear()
                continue

            ballBox: BoundingBox = ballInfo["bbox"]  # type: ignore[assignment]
            ballCenter: Point = get_center_of_bbox(ballBox)
            bestPid = self._bestCandidate(ballCenter, playerTracks[f], ballBox)

            if bestPid != -1:
                consecutive[bestPid] = consecutive.get(bestPid, 0) + 1
                if consecutive[bestPid] >= self.minFrames:
                    possessionList[f] = bestPid
                consecutive = {bestPid: consecutive[bestPid]}  # keep only current
            else:
                consecutive.clear()

        return possessionList
