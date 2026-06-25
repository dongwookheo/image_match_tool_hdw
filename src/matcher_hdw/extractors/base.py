from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseFeatureExtractor(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def extract(self, image: np.ndarray) -> dict:
        pass
