from __future__ import annotations

from importlib import import_module
from typing import Any

from .pipelines.sparse_pipeline import SparseMatchingPipeline
from .schema import ImageMatchingConfig

REGISTRY = {
    "sift_nn": {
        "type": "sparse",
        "extractor": "matcher_hdw.extractors.sift:SIFTExtractor",
        "matcher": "matcher_hdw.matchers.nearest_neighbor:NearestNeighborMatcher",
    },
}


def build_pipeline(config: ImageMatchingConfig) -> Any:
    spec = REGISTRY.get(config.algorithm)
    if spec is None:
        raise ValueError(f"unknown matcher algorithm: {config.algorithm}")

    pipeline_type = spec["type"]
    if pipeline_type == "sparse":
        extractor = _load_class(spec["extractor"])(config)
        matcher = _load_class(spec["matcher"])(config)
        return SparseMatchingPipeline(extractor, matcher, config)

    raise ValueError(f"unsupported pipeline type: {pipeline_type}")


def _load_class(path: str):
    module_name, class_name = path.split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, class_name)
