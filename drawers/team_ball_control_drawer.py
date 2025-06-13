
import cv2
import numpy as np
from typing import List, Sequence, Dict, Any

Frame = np.ndarray
AssignmentFrame = Dict[int, int]


class TeamBallControlDrawer:
    def __init__(self) -> None:
        pass

    def _teamControlArray(
        self,
        playerAssignment: Sequence[AssignmentFrame],
        ballAquisition: Sequence[int],
    ) -> np.ndarray:
        control: List[int] = []
        for assign, pid in zip(playerAssignment, ballAquisition):
            if pid == -1 or pid not in assign:
                control.append(-1)
            else:
                control.append(1 if assign[pid] == 1 else 2)
        return np.array(control)

    def _drawFrame(
        self,
        frame: Frame,
        idx: int,
        control: np.ndarray,
        fontScale: float = 0.7,
        thickness: int = 2,
    ) -> Frame:
        h, w = frame.shape[:2]
        rx1, ry1 = int(w * 0.60), int(h * 0.75)
        rx2, ry2 = int(w * 0.99), int(h * 0.90)
        tx = int(w * 0.63)
        ty1, ty2 = int(h * 0.80), int(h * 0.88)
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        seq = control[: idx + 1]
        t1 = (seq == 1).sum() / len(seq) if len(seq) else 0.0
        t2 = (seq == 2).sum() / len(seq) if len(seq) else 0.0
        cv2.putText(
            frame,
            f"Team 1 Ball Control: {t1 * 100:.2f}%",
            (tx, ty1),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            (0, 0, 0),
            thickness,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {t2 * 100:.2f}%",
            (tx, ty2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            (0, 0, 0),
            thickness,
        )
        return frame

    def draw(
        self,
        videoFrames: Sequence[Frame],
        playerAssignment: Sequence[AssignmentFrame],
        ballAquisition: Sequence[int],
    ) -> List[Frame]:
        control = self._teamControlArray(playerAssignment, ballAquisition)
        out: List[Frame] = []
        for idx, frame in enumerate(videoFrames):
            if idx == 0:
                out.append(frame.copy())
                continue
            out.append(self._drawFrame(frame.copy(), idx, control))
        return out
