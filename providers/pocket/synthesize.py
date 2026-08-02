from pocket_tts import TTSModel

from ..base import ProviderInfo

info = ProviderInfo(
    cloning=True,
    languages=["en"],
    # anna, azelma, charles, eve, fantine, george, jane, juergen, marius,
    # michael, paul, vera dropped: their reference recordings themselves
    # contain breath sounds, so the model reproduces it regardless of noise_clamp.
    presets=["alba", "bill_boerst", "caro_davy", "cosette", "eponine", "jean"],
)

_model = None
# ponytail: get_state_for_audio_prompt is documented as slow, cache per voice so
# repeated /tts calls for the same preset/ref_audio don't redo it every time.
_voice_states = {}


def _get_model():
    global _model
    if _model is None:
        # ponytail: flow-matching decoder samples unclamped Gaussian noise per
        # step (noise_clamp=None); rare tail draws trigger the model's learned
        # breath-sound mode. noise_clamp=1.7 alone still let breaths through in
        # listening tests; clamping tighter (0.6) plus more ODE decode steps
        # (4 vs default 1) removed them. ~3x slower to generate than default.
        _model = TTSModel.load_model(
            noise_clamp=0.6, lsd_decode_steps=4, temp=0.6, eos_threshold=-3.5
        )
        # ponytail: pocket_tts degrades on short inputs; the lib's own fix is
        # padding with leading spaces to bump token count, just off by default
        _model.pad_with_spaces_for_short_inputs = True
    return _model


def synthesize(text: str, **options) -> tuple:
    model = _get_model()
    voice = options.get("preset") or options.get("ref_audio") or "alba"
    if voice not in _voice_states:
        _voice_states[voice] = model.get_state_for_audio_prompt(voice)
    audio = model.generate_audio(_voice_states[voice], text)
    return audio.numpy(), model.sample_rate
