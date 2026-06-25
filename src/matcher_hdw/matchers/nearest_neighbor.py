from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFeatureMatcher


class NearestNeighborMatcher(BaseFeatureMatcher):
    def __init__(self, config):
        super().__init__(config)
        self.norm = _cv2_norm(config.matcher.norm)

    def match(self, features0: dict, features1: dict) -> dict:
        desc0 = features0["descriptors"]
        desc1 = features1["descriptors"]
        if len(desc0) == 0 or len(desc1) == 0:
            return _empty_matches()

        matcher = cv2.BFMatcher(self.norm, crossCheck=False)
        candidates = matcher.knnMatch(desc0, desc1, k=2)
        reverse = _reverse_best(desc0, desc1, self.norm, self.config.matcher.mutual_check)
        return _ratio_matches(candidates, reverse, self.config.matcher)


def _ratio_matches(candidates, reverse: dict[int, int], config) -> dict:
    matches0, matches1, scores = [], [], []
    for pair in candidates:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance >= config.ratio_threshold * second.distance:
            continue
        if reverse and reverse.get(first.trainIdx) != first.queryIdx:
            continue
        score = 1.0 / (first.distance + 1e-6)
        if score < config.match_threshold:
            continue
        matches0.append(first.queryIdx)
        matches1.append(first.trainIdx)
        scores.append(score)
    return {
        "matches0": np.asarray(matches0, dtype=np.int64),
        "matches1": np.asarray(matches1, dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float32),
    }


def _reverse_best(desc0: np.ndarray, desc1: np.ndarray, norm: int, enabled: bool) -> dict[int, int]:
    if not enabled:
        return {}
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    return {match.queryIdx: match.trainIdx for match in matcher.match(desc1, desc0)}


def _empty_matches() -> dict:
    empty_idx = np.empty((0,), dtype=np.int64)
    return {"matches0": empty_idx, "matches1": empty_idx, "scores": np.empty((0,), dtype=np.float32)}


def _cv2_norm(name: str) -> int:
    if name == "l2":
        return cv2.NORM_L2
    if name == "hamming":
        return cv2.NORM_HAMMING
    raise ValueError(f"unsupported matcher.norm: {name}")
