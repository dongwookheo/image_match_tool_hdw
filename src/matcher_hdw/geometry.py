from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .schema import GeometryConfig


@dataclass(frozen=True)
class GeometryResult:
    H_0_to_1: np.ndarray | None
    fundamental: np.ndarray | None
    inlier_mask: np.ndarray | None
    message: str


def estimate_geometry(
    mkpts0: np.ndarray, mkpts1: np.ndarray, config: GeometryConfig
) -> GeometryResult:
    pts0 = _as_points(mkpts0)
    pts1 = _as_points(mkpts1)
    if config.type == "none":
        return GeometryResult(None, None, np.ones(len(pts0), dtype=bool), "ok")
    if len(pts0) != len(pts1):
        return GeometryResult(None, None, None, "matched point counts differ")
    if len(pts0) < _min_points(config.type):
        return GeometryResult(None, None, None, "not enough matches for geometry")

    method = _cv2_method(config.ransac_method)
    if config.type == "homography":
        return _estimate_homography(pts0, pts1, method, config)
    if config.type == "fundamental":
        return _estimate_fundamental(pts0, pts1, method, config)
    return GeometryResult(None, None, None, f"unsupported geometry: {config.type}")


def _estimate_homography(
    pts0: np.ndarray, pts1: np.ndarray, method: int, config: GeometryConfig
) -> GeometryResult:
    try:
        matrix, mask = cv2.findHomography(
            pts0,
            pts1,
            method=method,
            ransacReprojThreshold=config.reproj_threshold,
            maxIters=config.max_iter,
            confidence=config.confidence,
        )
    except cv2.error as exc:
        return GeometryResult(None, None, None, f"homography failed: {exc}")
    return GeometryResult(matrix, None, _mask_to_bool(mask), "ok")


def _estimate_fundamental(
    pts0: np.ndarray, pts1: np.ndarray, method: int, config: GeometryConfig
) -> GeometryResult:
    try:
        matrix, mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            method=method,
            ransacReprojThreshold=config.reproj_threshold,
            confidence=config.confidence,
            maxIters=config.max_iter,
        )
    except cv2.error as exc:
        return GeometryResult(None, None, None, f"fundamental failed: {exc}")
    return GeometryResult(None, matrix, _mask_to_bool(mask), "ok")


def _as_points(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def _mask_to_bool(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    return mask.reshape(-1).astype(bool)


def _min_points(geometry_type: str) -> int:
    return 8 if geometry_type == "fundamental" else 4


def _cv2_method(name: str) -> int:
    methods = {
        "opencv": cv2.RANSAC,
        "usac_magsac": cv2.USAC_MAGSAC,
        "usac_default": cv2.USAC_DEFAULT,
    }
    return methods[name]
