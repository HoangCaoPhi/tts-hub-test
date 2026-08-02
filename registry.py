import json
from pathlib import Path

_ROOT = Path(__file__).parent
_PATH = _ROOT / "voices.json"


def _load() -> dict:
    if not _PATH.exists():
        return {}
    return json.loads(_PATH.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    _PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_all() -> list[dict]:
    return [
        {"voice_id": voice_id, "provider": provider, **entry}
        for voice_id, providers in _load().items()
        for provider, entry in providers.items()
    ]


def get(voice_id: str, provider: str) -> dict | None:
    entry = _load().get(voice_id, {}).get(provider)
    ref_audio = entry.get("options", {}).get("ref_audio") if entry else None
    if ref_audio and not Path(ref_audio).is_absolute():
        entry = {**entry, "options": {**entry["options"], "ref_audio": str(_ROOT / ref_audio)}}
    return entry


def put(voice_id: str, provider: str, entry: dict) -> dict:
    data = _load()
    data.setdefault(voice_id, {})[provider] = entry
    _save(data)
    return {"voice_id": voice_id, "provider": provider, **entry}
