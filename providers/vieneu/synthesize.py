import logging
import os
import re

import numpy as np
import torch
from vieneu import Vieneu

from ..base import ProviderInfo

logger = logging.getLogger("Vieneu.Synthesize")

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
_tts_v3 = None

# vieneu reads "/" as "trên" (fraction reading, e.g. "1/2" -> "một trên hai").
# Only replace when neither side is a digit, so real fractions still read correctly.
_SEPARATOR_SLASH = re.compile(r"(?<!\d)/(?!\d)")

# Map human-readable voice names to preset keys
VOICE_NAME_MAP = {
    "Phạm Tuyên": "Tuyen",
    "Xuân Vĩnh": "Vinh",
    "Thanh Bình": "Binh",
    "Thái Sơn": "Sơn",
    "Trúc Ly": "Ly",
    "Thục Đoan": "Doan",
    "Bích Ngọc": "Ngoc",
}


def _get_tts():
    global _tts
    if _tts is None:
        device = "cuda" if torch.cuda.is_available() else "auto"
        dtype = "float16" if torch.cuda.is_available() else "auto"
        _tts = Vieneu(mode="v3turbo", device=device, dtype=dtype)
        try:
            _tts.infer("Xin chào", temperature=0.35, top_k=20, max_chars=180)
        except Exception:
            pass
    return _tts


def synthesize(text: str, **options) -> tuple[np.ndarray, int]:
    text = _SEPARATOR_SLASH.sub(" hoặc ", text).strip()
    if text and not text.endswith((".", "!", "?", ";", ":")):
        text += "."

    tts = _get_tts()

    voice_kwargs = {}
    if "ref_audio" in options:
        voice_kwargs = {
            "ref_audio": options["ref_audio"],
            "ref_text": options.get("ref_text"),
        }
    else:
        preset_voice = options.get("voice_id") or options.get("voice") or options.get("preset") or "Tuyen"
        voice_kwargs = {"voice": preset_voice}

    audio = tts.infer(
        text,
        style=options.get("style", "tu_nhien"),
        temperature=float(options.get("temperature", 0.35)),
        top_k=int(options.get("top_k", 20)),
        top_p=float(options.get("top_p", 0.85)),
        repetition_penalty=float(options.get("repetition_penalty", 1.2)),
        max_chars=int(options.get("max_chars", 180)),
        **voice_kwargs,
    )

    return audio, tts.sample_rate

