from typing import Dict, List, Sequence

AssignmentFrame = Dict[int, int]


class PassAndInterceptionDetector:
    def __init__(self) -> None:
        pass

    def detectPasses(
        self,
        ballAcquisition: Sequence[int],
        playerAssignment: Sequence[AssignmentFrame],
    ) -> List[int]:
        passes = [-1] * len(ballAcquisition)
        prevHolder = -1
        prevFrame = -1
        for idx in range(1, len(ballAcquisition)):
            if ballAcquisition[idx - 1] != -1:
                prevHolder = ballAcquisition[idx - 1]
                prevFrame = idx - 1
            currHolder = ballAcquisition[idx]
            if prevHolder != -1 and currHolder != -1 and prevHolder != currHolder:
                prevTeam = playerAssignment[prevFrame].get(prevHolder, -1)
                currTeam = playerAssignment[idx].get(currHolder, -1)
                if prevTeam == currTeam and prevTeam != -1:
                    passes[idx] = prevTeam
        return passes

    def detectInterceptions(
        self,
        ballAcquisition: Sequence[int],
        playerAssignment: Sequence[AssignmentFrame],
    ) -> List[int]:
        interceptions = [-1] * len(ballAcquisition)
        prevHolder = -1
        prevFrame = -1
        for idx in range(1, len(ballAcquisition)):
            if ballAcquisition[idx - 1] != -1:
                prevHolder = ballAcquisition[idx - 1]
                prevFrame = idx - 1
            currHolder = ballAcquisition[idx]
            if prevHolder != -1 and currHolder != -1 and prevHolder != currHolder:
                prevTeam = playerAssignment[prevFrame].get(prevHolder, -1)
                currTeam = playerAssignment[idx].get(currHolder, -1)
                if prevTeam != currTeam and prevTeam != -1 and currTeam != -1:
                    interceptions[idx] = currTeam
        return interceptions
