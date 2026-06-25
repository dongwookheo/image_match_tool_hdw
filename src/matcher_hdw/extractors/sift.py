from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFeatureExtractor

SIFT_DESCRIPTOR_SIZE = 128
SIFT_PARAM_KEYS = {"contrast_threshold", "edge_threshold", "n_octave_layers", "sigma"}
SIFT_DEFAULT_CONTRAST_THRESHOLD = 0.04
SIFT_DEFAULT_EDGE_THRESHOLD = 10.0
SIFT_DEFAULT_N_OCTAVE_LAYERS = 3
SIFT_DEFAULT_SIGMA = 1.6


class SIFTExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__(config)
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("OpenCV SIFT_create is not available")
        params = _sift_params(config.feature.params)
        self.detector = cv2.SIFT_create(
            nfeatures=config.feature.max_keypoints,
            nOctaveLayers=params["n_octave_layers"],
            contrastThreshold=params["contrast_threshold"],
            edgeThreshold=params["edge_threshold"],
            sigma=params["sigma"],
        )

    def extract(self, image: np.ndarray) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if not keypoints or descriptors is None:
            return _empty_features()

        points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)
        scores = np.array(
            [keypoint.response for keypoint in keypoints], dtype=np.float32
        )
        return {
            "keypoints": points,
            "descriptors": descriptors.astype(np.float32, copy=False),
            "scores": scores,
        }


def _sift_params(raw_params: dict) -> dict:
    unknown = sorted(set(raw_params) - SIFT_PARAM_KEYS)
    if unknown:
        keys = ", ".join(unknown)
        raise ValueError(f"unsupported SIFT params: {keys}")

    params = {
        "contrast_threshold": _float_param(
            raw_params, "contrast_threshold", SIFT_DEFAULT_CONTRAST_THRESHOLD
        ),
        "edge_threshold": _float_param(
            raw_params, "edge_threshold", SIFT_DEFAULT_EDGE_THRESHOLD
        ),
        "n_octave_layers": _int_param(
            raw_params, "n_octave_layers", SIFT_DEFAULT_N_OCTAVE_LAYERS
        ),
        "sigma": _float_param(raw_params, "sigma", SIFT_DEFAULT_SIGMA),
    }
    _validate_sift_params(params)
    return params


def _float_param(raw_params: dict, name: str, default: float) -> float:
    return float(raw_params.get(name, default))


def _int_param(raw_params: dict, name: str, default: int) -> int:
    return int(raw_params.get(name, default))


def _validate_sift_params(params: dict) -> None:
    if params["contrast_threshold"] <= 0.0:
        raise ValueError("SIFT contrast_threshold must be positive")
    if params["edge_threshold"] <= 0.0:
        raise ValueError("SIFT edge_threshold must be positive")
    if params["n_octave_layers"] <= 0:
        raise ValueError("SIFT n_octave_layers must be positive")
    if params["sigma"] <= 0.0:
        raise ValueError("SIFT sigma must be positive")


def _empty_features() -> dict:
    return {
        "keypoints": np.empty((0, 2), dtype=np.float32),
        "descriptors": np.empty((0, SIFT_DESCRIPTOR_SIZE), dtype=np.float32),
        "scores": np.empty((0,), dtype=np.float32),
    }
