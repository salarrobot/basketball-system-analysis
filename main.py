from __future__ import annotations

import argparse
import os
from typing import Any, List

from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer,
)
from configs import (
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH,
)


def parseArgs() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input_video", type=str)
    p.add_argument("--output_video", type=str, default=OUTPUT_VIDEO_PATH)
    p.add_argument("--stub_path", type=str, default=STUBS_DEFAULT_PATH)
    return p.parse_args()


def main() -> None:
    a = parseArgs()
    frames = read_video(a.input_video)
    playerTracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ballTracker = BallTracker(BALL_DETECTOR_PATH)
    kpDetector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)
    playerTracks = playerTracker.objectTracks(
        frames,
        readFromStub=True,
        stubPath=os.path.join(a.stub_path, "player_track_stubs.pkl"),
    )
    ballTracks = ballTracker.objectTracks(
        frames,
        readFromStub=True,
        stubPath=os.path.join(a.stub_path, "ball_track_stubs.pkl"),
    )
    kpPerFrame = kpDetector.getCourtKeypoints(
        frames,
        readFromStub=True,
        stubPath=os.path.join(a.stub_path, "court_key_points_stub.pkl"),
    )
    ballTracks = ballTracker.removeWrongDetections(ballTracks)
    ballTracks = ballTracker.interpolateBallPositions(ballTracks)
    teamAssigner = TeamAssigner()
    playerAssignment = teamAssigner.assignTeams(
        frames,
        playerTracks,
        readFromStub=True,
        stubPath=os.path.join(a.stub_path, "player_assignment_stub.pkl"),
    )
    baDetector = BallAquisitionDetector()
    ballAquisition = baDetector.detectBallPossession(playerTracks, ballTracks)
    piDetector = PassAndInterceptionDetector()
    passes = piDetector.detectPasses(ballAquisition, playerAssignment)
    interceptions = piDetector.detectInterceptions(ballAquisition, playerAssignment)
    tvc = TacticalViewConverter("./images/basketball_court.png")
    kpPerFrame = tvc.validateKeypoints(kpPerFrame)
    tacticalPos = tvc.transformPlayers(kpPerFrame, playerTracks)
    sdc = SpeedAndDistanceCalculator(
        tvc.width,
        tvc.height,
        tvc.actualWidthM,
        tvc.actualHeightM,
    )
    distances = sdc.calculate_distance(tacticalPos)
    speeds = sdc.calculate_speed(distances)
    pDrawer = PlayerTracksDrawer()
    bDrawer = BallTracksDrawer()
    kpDrawer = CourtKeypointDrawer()
    tbDrawer = TeamBallControlDrawer()
    fnDrawer = FrameNumberDrawer()
    piDrawer = PassInterceptionDrawer()
    tvDrawer = TacticalViewDrawer()
    sdDrawer = SpeedAndDistanceDrawer()
    outFrames: List[Any] = pDrawer.draw(frames, playerTracks, playerAssignment, ballAquisition)
    outFrames = bDrawer.draw(outFrames, ballTracks)
    outFrames = kpDrawer.draw(outFrames, kpPerFrame)
    outFrames = fnDrawer.drawNumbers(outFrames)
    outFrames = tbDrawer.draw(outFrames, playerAssignment, ballAquisition)
    outFrames = piDrawer.draw(outFrames, passes, interceptions)
    outFrames = sdDrawer.drawMetrics(outFrames, playerTracks, distances, speeds)
    outFrames = tvDrawer.draw(
        outFrames,
        tvc.courtImagePath,
        tvc.width,
        tvc.height,
        tvc.keyPoints,
        tacticalPos,
        playerAssignment,
        ballAquisition,
    )
    save_video(outFrames, a.output_video)


if __name__ == "__main__":
    main()
