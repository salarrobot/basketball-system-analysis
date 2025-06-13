from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

folder = Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder, "../"))
from utils import read_stub, save_stub  # type: ignore


class TeamAssigner:
    def __init__(self, team1Class: str = "white shirt", team2Class: str = "dark blue shirt") -> None:
        self.playerTeam: Dict[int, int] = {}
        self.team1Class = team1Class
        self.team2Class = team2Class
        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None

    def _loadModel(self) -> None:
        if self.model is None or self.processor is None:
            self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def _playerColor(self, frame: Any, bbox: Sequence[float]) -> str:
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        classes = [self.team1Class, self.team2Class]
        inp = self.processor(text=classes, images=img, return_tensors="pt", padding=True)
        logits = self.model(**inp).logits_per_image
        return classes[int(logits.softmax(dim=1).argmax())]

    def _teamForPlayer(self, frame: Any, bbox: Sequence[float], pid: int) -> int:
        if pid in self.playerTeam:
            return self.playerTeam[pid]
        color = self._playerColor(frame, bbox)
        team = 1 if color == self.team1Class else 2
        self.playerTeam[pid] = team
        return team

    def assignTeams(
        self,
        videoFrames: Sequence[Any],
        playerTracks: Sequence[Dict[int, Dict[str, Any]]],
        *,
        readFromStub: bool = False,
        stubPath: str | None = None,
    ) -> List[Dict[int, int]]:
        cached = read_stub(readFromStub, stubPath)
        if cached is not None and len(cached) == len(videoFrames):
            return cached
        self._loadModel()
        res: List[Dict[int, int]] = []
        for idx, tracks in enumerate(playerTracks):
            if idx % 50 == 0:
                self.playerTeam.clear()
            frameRes: Dict[int, int] = {}
            for pid, info in tracks.items():
                frameRes[pid] = self._teamForPlayer(videoFrames[idx], info["bbox"], pid)
            res.append(frameRes)
        save_stub(stubPath, res)
        return res
