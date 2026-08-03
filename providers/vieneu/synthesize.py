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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # Load Standard Mode PyTorch GPU (VieNeu-TTS-v2 10,000h model, bypasses GGUF/llama-cpp)
            _tts = Vieneu(
                mode="standard",
                backbone_repo="pnnbao-ump/VieNeu-TTS-v2",
                gguf_filename=None,
                backbone_device=device,
                codec_device=device,
            )
        except Exception:
            # Fallback to v3turbo if standard PyTorch backend fails
            dtype = "float16" if torch.cuda.is_available() else "auto"
            _tts = Vieneu(mode="v3turbo", device=device, dtype=dtype)
        # Warmup GPU
        try:
            _tts.infer("Xin chào", temperature=0.3)
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
        if preset_voice:
            voice_kwargs = {"voice": preset_voice}

    # Ultra-stable settings for VieNeu v2 Standard
    temperature = float(options.get("temperature", 0.30))
    top_k = int(options.get("top_k", 20))

    # Standard engine supports inference kwargs
    try:
        audio = tts.infer(
            text,
            temperature=temperature,
            top_k=top_k,
            max_chars=int(options.get("max_chars", 200)),
            **voice_kwargs,
        )
    except TypeError:
        audio = tts.infer(text, **voice_kwargs)

    return audio, tts.sample_rate

