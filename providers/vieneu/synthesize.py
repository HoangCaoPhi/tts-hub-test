import os
import re
import tempfile

import soundfile as sf
from llama_cpp import Llama
from vieneu import Vieneu

from ..base import ProviderInfo

# ponytail: vieneu's _infer_ggml calls the llama_cpp backbone without a
# repeat_penalty, so it defaults to 1.0 (off) — nothing discourages the model
# from looping on a token pattern instead of emitting the stop token, which is
# the same EOS-skip behind "duration anomaly" retries upstream. Patched at the
# class level (not per-call) since vieneu's public API has no pass-through for
# it. >1.0 works process-wide since only VieNeu uses llama_cpp here.
_ORIGINAL_LLAMA_CALL = Llama.__call__


def _llama_call_with_repeat_penalty(self, *args, **kwargs):
    kwargs.setdefault("repeat_penalty", 1.15)
    return _ORIGINAL_LLAMA_CALL(self, *args, **kwargs)


Llama.__call__ = _llama_call_with_repeat_penalty

info = ProviderInfo(
    cloning=True,
    languages=["vi"],
    presets=["Binh", "Tuyen", "Vinh", "Doan", "Ly", "Sơn", "Ngoc"],
)

_tts = None
_tts_clone = None

# ponytail: vieneu reads "/" as "trên" (fraction reading, e.g. "1/2" -> "một
# trên hai"). Correct for actual fractions/dates but wrong when "/" is just a
# separator between words (e.g. "A/B"). Only replace when neither side is a
# digit, so real fractions still read correctly.
_SEPARATOR_SLASH = re.compile(r"(?<!\d)/(?!\d)")


# ponytail: generation is autoregressive sampling and occasionally never emits
# the EOS token, running away to the max_context cap instead of stopping — the
# "duration anomaly" retries in downstream callers. Capping max_context here
# (default 2048) makes each runaway fail in ~half the time instead of ~41s of
# wasted CPU inference per chunk (chunks are already split at <=256 chars).
_MAX_CONTEXT = 1024


def _get_tts():
    global _tts
    if _tts is None:
        _tts = Vieneu(mode="standard", backbone_device="cuda", codec_device="cuda")
        _tts.max_context = _MAX_CONTEXT
    return _tts


def _get_tts_clone():
    # ponytail: the default ONNX codec (neucodec-onnx-decoder-int8) only
    # decodes, so ref_audio cloning (which needs encode_code) crashes on it.
    # Cloning needs a torch codec instead, loaded separately and only on first
    # actual clone request so preset-only calls stay on the light path.
    # Must be "distill-neucodec", not the full "neucodec" — that's what
    # vieneu's own fast/remote/xpu modes default to for the trained backbone;
    # the full codec produces codes the backbone was never trained on and
    # generation runs away into incoherent noise instead of stopping at EOS.
    global _tts_clone
    if _tts_clone is None:
        _tts_clone = Vieneu(mode="standard", codec_repo="neuphonic/distill-neucodec", backbone_device="cuda", codec_device="cuda")
        _tts_clone.max_context = _MAX_CONTEXT
    return _tts_clone


def synthesize(text: str, **options) -> tuple:
    text = _SEPARATOR_SLASH.sub(" hoặc ", text)

    if "ref_audio" in options:
        tts = _get_tts_clone()
        voice_kwargs = {"ref_audio": options["ref_audio"], "ref_text": options["ref_text"]}
    else:
        tts = _get_tts()
        voice_kwargs = {"voice": tts.get_preset_voice(options.get("preset", "Doan"))}

    # ponytail: default temperature=1.0/top_k=50 let prosody/pronunciation
    # drift between runs; lower values trade some naturalness for steadier
    # output and fewer EOS-skip runaways ("duration anomaly" retries upstream).
    audio = tts.infer(
        text,
        temperature=options.get("temperature", 0.3),
        top_k=options.get("top_k", 10),
        **voice_kwargs,
    )

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        tts.save(audio, path)
        data, sample_rate = sf.read(path)
    finally:
        os.remove(path)
    return data, sample_rate
