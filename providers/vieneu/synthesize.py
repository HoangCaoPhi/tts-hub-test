import logging
import os
import re

import numpy as np
import torch
from vieneu import Vieneu

from ..base import ProviderInfo

# Enable CUDA optimizations if available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

info = ProviderInfo(
    cloning=True,
    languages=["vi"],
    presets=["Binh", "Tuyen", "Vinh", "Doan", "Ly", "Sơn", "Ngoc"],
)

_tts = None

# vieneu reads "/" as "trên" (fraction reading, e.g. "1/2" -> "một trên hai").
# Only replace when neither side is a digit, so real fractions still read correctly.
_SEPARATOR_SLASH = re.compile(r"(?<!\d)/(?!\d)")


# Map friendly IDs and full Vietnamese names to exact v3turbo preset keys
VOICE_MAPPING = {
    "Tuyen": "Phạm Tuyên",
    "Phạm Tuyên": "Phạm Tuyên",
    "Vinh": "Xuân Vĩnh",
    "Xuân Vĩnh": "Xuân Vĩnh",
    "Binh": "Thanh Bình",
    "Thanh Bình": "Thanh Bình",
    "Son": "Thái Sơn",
    "Sơn": "Thái Sơn",
    "Thái Sơn": "Thái Sơn",
    "Doan": "Quốc Đoàn",
    "Quốc Đoàn": "Quốc Đoàn",
    "Ly": "Trúc Ly",
    "Trúc Ly": "Trúc Ly",
    "Ngoc": "Bích Ngọc",
    "Bích Ngọc": "Bích Ngọc",
}


def _get_tts():
    global _tts
    if _tts is None:
        device = "cuda" if torch.cuda.is_available() else "auto"
        dtype = "float16" if torch.cuda.is_available() else "auto"
        _tts = Vieneu(mode="v3turbo", device=device, dtype=dtype)
        # Warmup GPU
        try:
            _tts.infer("Xin chào", temperature=0.50, top_k=40, max_chars=180)
        except Exception:
            pass
    return _tts


def synthesize(text: str, **options) -> tuple[np.ndarray, int]:
    text = _SEPARATOR_SLASH.sub(" hoặc ", text)
    tts = _get_tts()

    voice_kwargs = {}
    if "ref_audio" in options:
        voice_kwargs = {
            "ref_audio": options["ref_audio"],
            "ref_text": options.get("ref_text"),
            "denoise": options.get("denoise", True),
        }
    else:
        raw_voice = options.get("voice_id") or options.get("voice") or options.get("preset") or "Phạm Tuyên"
        preset_voice = VOICE_MAPPING.get(raw_voice, raw_voice)
        if preset_voice:
            voice_kwargs = {"voice": preset_voice}

    # Fluent parameters for VieNeu v3Turbo
    audio = tts.infer(
        text,
        style=options.get("style", "tu_nhien"),
        temperature=float(options.get("temperature", 0.50)),
        top_k=int(options.get("top_k", 40)),
        top_p=float(options.get("top_p", 0.90)),
        repetition_penalty=float(options.get("repetition_penalty", 1.15)),
        max_chars=int(options.get("max_chars", 180)),
        **voice_kwargs,
    )

    return audio, tts.sample_rate
