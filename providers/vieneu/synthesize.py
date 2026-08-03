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


def _get_tts():
    global _tts
    if _tts is None:
        device = "cuda" if torch.cuda.is_available() else "auto"
        dtype = "float16" if torch.cuda.is_available() else "auto"
        _tts = Vieneu(mode="v3turbo", device=device, dtype=dtype)
        # Warmup GPU
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
            "denoise": options.get("denoise", True),
        }
    else:
        preset_voice = options.get("voice_id") or options.get("voice") or options.get("preset") or "Tuyen"
        if preset_voice:
            voice_kwargs = {"voice": preset_voice}

    # Dynamic temperature based on length: <= 4 words gets 0.5 for better natural prosody, longer gets 0.35
    num_words = len(text.split())
    default_temp = 0.5 if num_words <= 4 else 0.35
    temperature = float(options.get("temperature", default_temp))

    # Optimal hyperparameter defaults for high voice stability & prosody on GPU
    audio = tts.infer(
        text,
        style=options.get("style", "tu_nhien"),
        temperature=temperature,
        top_k=int(options.get("top_k", 20)),
        top_p=float(options.get("top_p", 0.85)),
        repetition_penalty=float(options.get("repetition_penalty", 1.2)),
        max_chars=int(options.get("max_chars", 180)),
        **voice_kwargs,
    )

    return audio, tts.sample_rate

