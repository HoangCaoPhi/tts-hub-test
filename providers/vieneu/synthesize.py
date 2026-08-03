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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # Load Standard Mode PyTorch GPU (VieNeu-TTS-v2 10,000h model)
            _tts = Vieneu(
                mode="standard",
                backbone_repo="pnnbao-ump/VieNeu-TTS-v2",
                gguf_filename=None,
                backbone_device=device,
                codec_device=device,
            )
        except Exception as e:
            logger.warning(f"Standard mode init failed ({e}), falling back to v3turbo")
            dtype = "float16" if torch.cuda.is_available() else "auto"
            _tts = Vieneu(mode="v3turbo", device=device, dtype=dtype)
        # Warmup GPU
        try:
            _tts.infer("Xin chào", temperature=0.3)
        except Exception:
            pass
    return _tts


def _get_tts_v3():
    global _tts_v3
    if _tts_v3 is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "float16" if torch.cuda.is_available() else "auto"
        _tts_v3 = Vieneu(mode="v3turbo", device=device, dtype=dtype)
    return _tts_v3


def synthesize(text: str, **options) -> tuple[np.ndarray, int]:
    text = _SEPARATOR_SLASH.sub(" hoặc ", text).strip()
    if text and not text.endswith((".", "!", "?", ";", ":")):
        text += "."

    raw_voice = options.get("voice_id") or options.get("voice") or options.get("preset") or "Tuyen"

    try:
        tts = _get_tts()
        voice_kwargs = {}
        if "ref_audio" in options:
            voice_kwargs = {
                "ref_audio": options["ref_audio"],
                "ref_text": options.get("ref_text"),
            }
        else:
            preset_voice = VOICE_NAME_MAP.get(raw_voice, raw_voice)
            if hasattr(tts, "_preset_voices") and tts._preset_voices:
                if preset_voice not in tts._preset_voices:
                    preset_voice = getattr(tts, "_default_voice", None) or next(iter(tts._preset_voices.keys()))
            if preset_voice:
                voice_kwargs = {"voice": preset_voice}

        temperature = float(options.get("temperature", 0.30))
        top_k = int(options.get("top_k", 20))

        audio = tts.infer(
            text,
            temperature=temperature,
            top_k=top_k,
            max_chars=int(options.get("max_chars", 200)),
            **voice_kwargs,
        )
        sr = getattr(tts, "sample_rate", 24000)
        return audio, sr
    except Exception as e:
        logger.warning(f"Standard infer failed ({e}), falling back to v3turbo")
        v3 = _get_tts_v3()
        v3_voice = VOICE_NAME_MAP.get(raw_voice, raw_voice)
        audio = v3.infer(
            text,
            voice=v3_voice,
            style=options.get("style", "tu_nhien"),
            temperature=0.35,
            top_k=20,
            max_chars=180,
        )
        return audio, v3.sample_rate

