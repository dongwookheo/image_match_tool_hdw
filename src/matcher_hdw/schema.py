from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SUPPORTED_GEOMETRIES = {"none", "homography", "fundamental"}
SUPPORTED_RANSAC_METHODS = {"opencv", "usac_magsac", "usac_default"}
SUPPORTED_RUNTIME_DEVICES = {"auto", "cpu", "cuda"}
FEATURE_KEYS = {"name", "max_keypoints", "params"}


@dataclass(frozen=True)
class FeatureConfig:
    name: str = "sift"
    max_keypoints: int = 3000
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherConfig:
    type: str = "nearest_neighbor"
    norm: str = "l2"
    ratio_threshold: float = 0.75
    mutual_check: bool = False
    match_threshold: float = 0.0


@dataclass(frozen=True)
class GeometryConfig:
    type: str = "homography"
    ransac_method: str = "usac_magsac"
    reproj_threshold: float = 3.0
    confidence: float = 0.995
    max_iter: int = 5000


@dataclass(frozen=True)
class QualityConfig:
    min_matches: int = 20
    min_inliers: int = 10
    min_inlier_ratio: float = 0.2


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "float32"
    cache_dir: str | None = ".cache/models"
    verbose: bool = True


@dataclass(frozen=True)
class ImageMatchingConfig:
    algorithm: str
    feature: FeatureConfig
    matcher: MatcherConfig
    geometry: GeometryConfig
    quality: QualityConfig
    runtime: RuntimeConfig


@dataclass
class MatchResult:
    ok: bool
    message: str
    image0_shape: tuple[int, int]
    image1_shape: tuple[int, int]
    keypoints0: np.ndarray
    keypoints1: np.ndarray
    matched_keypoints0: np.ndarray
    matched_keypoints1: np.ndarray
    scores: np.ndarray | None
    inlier_mask: np.ndarray | None
    inlier_keypoints0: np.ndarray | None
    inlier_keypoints1: np.ndarray | None
    H_0_to_1: np.ndarray | None
    fundamental: np.ndarray | None
    num_keypoints0: int
    num_keypoints1: int
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    raw: dict[str, Any]


def config_from_dict(raw: dict[str, Any]) -> ImageMatchingConfig:
    if not isinstance(raw, dict):
        raise TypeError("matcher config must be a mapping")

    algorithm = str(raw.get("algorithm", "")).strip()
    if not algorithm:
        raise ValueError("matcher config requires a non-empty algorithm")

    config = ImageMatchingConfig(
        algorithm=algorithm,
        feature=_feature_config(raw.get("feature", {})),
        matcher=_matcher_config(raw.get("matcher", {})),
        geometry=_geometry_config(raw.get("geometry", {})),
        quality=_quality_config(raw.get("quality", {})),
        runtime=_runtime_config(raw.get("runtime", {})),
    )
    _validate_config(config)
    return config


def _feature_config(raw: Any) -> FeatureConfig:
    data = _mapping(raw, "feature")
    _validate_keys(data, FEATURE_KEYS, "feature")
    params = _mapping(data.get("params", {}), "feature.params")
    return FeatureConfig(
        name=str(data.get("name", "sift")).lower(),
        max_keypoints=int(data.get("max_keypoints", 3000)),
        params=dict(params),
    )


def _matcher_config(raw: Any) -> MatcherConfig:
    data = _mapping(raw, "matcher")
    return MatcherConfig(
        type=str(data.get("type", "nearest_neighbor")).lower(),
        norm=str(data.get("norm", "l2")).lower(),
        ratio_threshold=float(data.get("ratio_threshold", 0.75)),
        mutual_check=bool(data.get("mutual_check", False)),
        match_threshold=float(data.get("match_threshold", 0.0)),
    )


def _geometry_config(raw: Any) -> GeometryConfig:
    data = _mapping(raw, "geometry")
    return GeometryConfig(
        type=str(data.get("type", "homography")).lower(),
        ransac_method=str(data.get("ransac_method", "usac_magsac")).lower(),
        reproj_threshold=float(data.get("reproj_threshold", 3.0)),
        confidence=float(data.get("confidence", 0.995)),
        max_iter=int(data.get("max_iter", 5000)),
    )


def _quality_config(raw: Any) -> QualityConfig:
    data = _mapping(raw, "quality")
    return QualityConfig(
        min_matches=int(data.get("min_matches", 20)),
        min_inliers=int(data.get("min_inliers", 10)),
        min_inlier_ratio=float(data.get("min_inlier_ratio", 0.2)),
    )


def _runtime_config(raw: Any) -> RuntimeConfig:
    data = _mapping(raw, "runtime")
    cache_dir = data.get("cache_dir", ".cache/models")
    return RuntimeConfig(
        device=str(data.get("device", "auto")).lower(),
        dtype=str(data.get("dtype", "float32")).lower(),
        cache_dir=None if cache_dir is None else str(cache_dir),
        verbose=bool(data.get("verbose", True)),
    )


def _mapping(raw: Any, name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"{name} section must be a mapping")
    return raw


def _validate_keys(data: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        keys = ", ".join(unknown)
        raise ValueError(f"unsupported {name} keys: {keys}")


def _validate_config(config: ImageMatchingConfig) -> None:
    if config.feature.max_keypoints <= 0:
        raise ValueError("feature.max_keypoints must be positive")
    if not 0.0 < config.matcher.ratio_threshold <= 1.0:
        raise ValueError("matcher.ratio_threshold must be in (0, 1]")
    if config.geometry.type not in SUPPORTED_GEOMETRIES:
        raise ValueError(f"unsupported geometry.type: {config.geometry.type}")
    if config.geometry.ransac_method not in SUPPORTED_RANSAC_METHODS:
        raise ValueError(f"unsupported ransac_method: {config.geometry.ransac_method}")
    if config.runtime.device not in SUPPORTED_RUNTIME_DEVICES:
        raise ValueError(f"unsupported runtime.device: {config.runtime.device}")
    if config.quality.min_inlier_ratio < 0.0:
        raise ValueError("quality.min_inlier_ratio must be non-negative")
