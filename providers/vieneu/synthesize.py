import os
import re
import tempfile

import soundfile as sf
from vieneu import Vieneu

from ..base import ProviderInfo

info = ProviderInfo(
    cloning=True,
    languages=["vi"],
    presets=["Binh", "Tuyen", "Vinh", "Doan", "Ly", "Sơn", "Ngoc"],
)

_tts = None

# ponytail: vieneu reads "/" as "trên" (fraction reading, e.g. "1/2" -> "một
# trên hai"). Correct for actual fractions/dates but wrong when "/" is just a
# separator between words (e.g. "A/B"). Only replace when neither side is a
# digit, so real fractions still read correctly.
_SEPARATOR_SLASH = re.compile(r"(?<!\d)/(?!\d)")


def _get_tts():
    global _tts
    if _tts is None:
        # Use v3turbo which defaults to PyTorch on GPU automatically
        _tts = Vieneu(mode="v3turbo")
    return _tts


def synthesize(text: str, **options) -> tuple:
    text = _SEPARATOR_SLASH.sub(" hoặc ", text)
    tts = _get_tts()
    voice_kwargs = {}

    if "ref_audio" in options:
        voice_kwargs = {"ref_audio": options["ref_audio"], "ref_text": options["ref_text"]}
    else:
        voice_kwargs = {"voice_name": options.get("voice_id")}

    # Default parameters for naturalness
    audio = tts.infer(
        text,
        temperature=options.get("temperature", 0.7),
        top_k=options.get("top_k", 40),
        **voice_kwargs,
    )

    _, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        sf.write(tmp_path, audio, tts.sample_rate)
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return audio_bytes, "audio/wav"
