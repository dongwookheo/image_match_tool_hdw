from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .cache import builtin_matcher_config_path, resolve_cache_dir
from .factory import build_pipeline
from .schema import ImageMatchingConfig, MatchResult, config_from_dict


def load_matcher_config(path: str | Path) -> ImageMatchingConfig:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return config_from_dict(raw)


def load_builtin_config(name: str) -> ImageMatchingConfig:
    return load_matcher_config(builtin_matcher_config_path(name))


class ImageMatcher:
    def __init__(self, config: ImageMatchingConfig):
        self.config = config
        self.cache_dir = resolve_cache_dir(config.runtime.cache_dir, create=False)
        self.pipeline = build_pipeline(config)

    def match(self, image0: Any, image1: Any) -> MatchResult:
        image0_bgr = _as_bgr_image(image0, "image0")
        image1_bgr = _as_bgr_image(image1, "image1")
        return self.pipeline.run(image0_bgr, image1_bgr)


def _as_bgr_image(image: Any, name: str) -> np.ndarray:
    if isinstance(image, (str, Path)):
        loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if loaded is None:
            raise ValueError(f"{name} could not be read: {image}")
        return loaded
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a path or numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be a BGR image with shape HxWx3")
    return image
