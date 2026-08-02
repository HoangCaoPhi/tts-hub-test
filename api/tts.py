import io

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import spaces
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

import registry
from providers import PROVIDERS

router = APIRouter()

_TARGET_LUFS = -16.0


def _normalize_loudness(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    if len(mono) / sample_rate < 0.5:
        return audio  # too short for pyloudnorm's measurement window, leave as-is
    loudness = pyln.Meter(sample_rate).integrated_loudness(mono.astype(np.float64))
    if not np.isfinite(loudness):
        return audio
    normalized = pyln.normalize.loudness(audio, loudness, _TARGET_LUFS)
    return np.clip(normalized, -1.0, 1.0)


class TtsRequest(BaseModel):
    provider: str
    text: str
    voice_id: str | None = None
    options: dict = {}


@router.post("/tts")
@spaces.GPU
def post_tts(body: TtsRequest):
    if body.provider not in PROVIDERS:
        raise HTTPException(400, f"unknown provider '{body.provider}'")

    options = dict(body.options)

    if body.voice_id:
        entry = registry.get(body.voice_id, body.provider)
        if entry is None:
            raise HTTPException(404, f"voice_id '{body.voice_id}' has no '{body.provider}' registration")
        options = {**entry.get("options", {}), **options}

    audio, sample_rate = PROVIDERS[body.provider](body.text, **options)
    audio = _normalize_loudness(audio, sample_rate)

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")
