from __future__ import annotations

import os
import sys
import pathlib
from typing import List, Dict, Tuple

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, "../"))
from utils import measure_distance  # type: ignore


class SpeedAndDistanceCalculator:
    def __init__(
        self,
        width_in_pixels: int,
        height_in_pixels: int,
        width_in_meters: float,
        height_in_meters: float,
    ) -> None:
        self.width_in_pixels = width_in_pixels
        self.height_in_pixels = height_in_pixels
        self.width_in_meters = width_in_meters
        self.height_in_meters = height_in_meters

    def calculate_distance(self, tactical_player_positions: List[Dict[int, Tuple[float, float]]]):
        previous_players_position: Dict[int, Tuple[float, float]] = {}
        output_distances: List[Dict[int, float]] = []
        for frame_positions in tactical_player_positions:
            output_distances.append({})
            for pid, curr_pos in frame_positions.items():
                if pid in previous_players_position:
                    prev_pos = previous_players_position[pid]
                    meter_dist = self._meter_distance(prev_pos, curr_pos)
                    output_distances[-1][pid] = meter_dist
                previous_players_position[pid] = curr_pos
        return output_distances

    def _meter_distance(
        self,
        prev_pixel_pos: Tuple[float, float],
        curr_pixel_pos: Tuple[float, float],
    ) -> float:
        px, py = prev_pixel_pos
        cx, cy = curr_pixel_pos
        prev_mx = px * self.width_in_meters / self.width_in_pixels
        prev_my = py * self.height_in_meters / self.height_in_pixels
        curr_mx = cx * self.width_in_meters / self.width_in_pixels
        curr_my = cy * self.height_in_meters / self.height_in_pixels
        meter_dist = measure_distance((curr_mx, curr_my), (prev_mx, prev_my))
        return meter_dist * 0.4

    def calculate_speed(
        self,
        distances: List[Dict[int, float]],
        fps: float = 30,
    ) -> List[Dict[int, float]]:
        speeds: List[Dict[int, float]] = []
        window = 5
        for idx in range(len(distances)):
            speeds.append({})
            for pid in distances[idx].keys():
                start = max(0, idx - (window * 3) + 1)
                total_dist = 0.0
                frames_present = 0
                last_idx = None
                for i in range(start, idx + 1):
                    if pid in distances[i]:
                        if last_idx is not None:
                            total_dist += distances[i][pid]
                            frames_present += 1
                        last_idx = i
                if frames_present >= window:
                    time_sec = frames_present / fps
                    hours = time_sec / 3600.0
                    speed_kmh = (total_dist / 1000.0) / hours if hours > 0 else 0.0
                    speeds[idx][pid] = speed_kmh
                else:
                    speeds[idx][pid] = 0.0
        return speeds
