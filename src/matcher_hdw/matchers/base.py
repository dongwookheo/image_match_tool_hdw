from __future__ import annotations

from abc import ABC, abstractmethod


class BaseFeatureMatcher(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def match(self, features0: dict, features1: dict) -> dict:
        pass
