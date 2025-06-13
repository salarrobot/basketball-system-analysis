from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .utils import draw_ellipse, draw_traingle

Frame = np.ndarray
PlayerInfo = Dict[str, Any]
TracksFrame = Dict[int, PlayerInfo]


class PlayerTracksDrawer:
    def __init__(
        self,
        team1Color: Tuple[int, int, int] = (255, 245, 238),
        team2Color: Tuple[int, int, int] = (128, 0, 0),
        defaultTeamId: int = 1,
    ) -> None:
        self.team1Color = team1Color
        self.team2Color = team2Color
        self.defaultTeamId = defaultTeamId

    def drawTracks(
        self,
        videoFrames: Sequence[Frame],
        tracks: Sequence[TracksFrame],
        playerAssignment: Sequence[Dict[int, int]],
        ballAquisition: Sequence[int],
    ) -> List[Frame]:
        output: List[Frame] = []
        for idx, frame in enumerate(videoFrames):
            f = frame.copy()
            pDict = tracks[idx]
            assignment = playerAssignment[idx]
            ballPid = ballAquisition[idx]
            for pid, player in pDict.items():
                teamId = assignment.get(pid, self.defaultTeamId)
                color = self.team1Color if teamId == 1 else self.team2Color
                f = draw_ellipse(f, player["bbox"], color, pid)
                if pid == ballPid:
                    f = draw_traingle(f, player["bbox"], (0, 0, 255))
            output.append(f)
        return output
