from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Synthesize(Protocol):
    def __call__(self, text: str, **options) -> tuple[np.ndarray, int]: ...


@dataclass
class ProviderInfo:
    cloning: bool | None
    languages: list[str]
    presets: list[str] | None
