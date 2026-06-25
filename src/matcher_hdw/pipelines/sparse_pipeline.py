from __future__ import annotations

import numpy as np

from matcher_hdw.geometry import estimate_geometry
from matcher_hdw.schema import ImageMatchingConfig, MatchResult


class SparseMatchingPipeline:
    def __init__(self, extractor, matcher, config: ImageMatchingConfig):
        self.extractor = extractor
        self.matcher = matcher
        self.config = config

    def run(self, image0: np.ndarray, image1: np.ndarray) -> MatchResult:
        features0 = self.extractor.extract(image0)
        features1 = self.extractor.extract(image1)
        match_data = self.matcher.match(features0, features1)
        matched0, matched1 = _matched_points(features0, features1, match_data)
        geometry = estimate_geometry(matched0, matched1, self.config.geometry)
        inlier_mask = geometry.inlier_mask
        inlier0, inlier1 = _inlier_points(matched0, matched1, inlier_mask)
        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        num_matches = len(matched0)
        ratio = num_inliers / num_matches if num_matches else 0.0
        ok, message = _quality_status(self.config, geometry, num_matches, num_inliers, ratio)
        return MatchResult(
            ok=ok,
            message=message,
            image0_shape=image0.shape[:2],
            image1_shape=image1.shape[:2],
            keypoints0=features0["keypoints"],
            keypoints1=features1["keypoints"],
            matched_keypoints0=matched0,
            matched_keypoints1=matched1,
            scores=match_data.get("scores"),
            inlier_mask=inlier_mask,
            inlier_keypoints0=inlier0,
            inlier_keypoints1=inlier1,
            H_0_to_1=geometry.H_0_to_1,
            fundamental=geometry.fundamental,
            num_keypoints0=len(features0["keypoints"]),
            num_keypoints1=len(features1["keypoints"]),
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=ratio,
            raw={"features0": features0, "features1": features1, "matches": match_data},
        )


def _matched_points(features0: dict, features1: dict, match_data: dict) -> tuple[np.ndarray, np.ndarray]:
    idx0 = match_data["matches0"]
    idx1 = match_data["matches1"]
    return features0["keypoints"][idx0], features1["keypoints"][idx1]


def _inlier_points(matched0: np.ndarray, matched1: np.ndarray, mask) -> tuple[np.ndarray | None, np.ndarray | None]:
    if mask is None:
        return None, None
    return matched0[mask], matched1[mask]


def _quality_status(config: ImageMatchingConfig, geometry, num_matches: int, num_inliers: int, ratio: float):
    quality = config.quality
    if num_matches < quality.min_matches:
        return False, f"not enough matches: {num_matches} < {quality.min_matches}"
    if config.geometry.type != "none" and geometry.inlier_mask is None:
        return False, geometry.message
    if config.geometry.type == "homography" and geometry.H_0_to_1 is None:
        return False, "homography was not estimated"
    if config.geometry.type == "fundamental" and geometry.fundamental is None:
        return False, "fundamental matrix was not estimated"
    if num_inliers < quality.min_inliers:
        return False, f"not enough inliers: {num_inliers} < {quality.min_inliers}"
    if ratio < quality.min_inlier_ratio:
        return False, f"inlier ratio too low: {ratio:.3f} < {quality.min_inlier_ratio:.3f}"
    return True, "ok"
