import json
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

import registry

router = APIRouter()

REPO_ROOT = Path(__file__).parent.parent
VOICES_DIR = REPO_ROOT / "voices"


@router.get("/voices")
def list_voices():
    return registry.list_all()


@router.post("/voices", status_code=201)
def add_voice(
    voice_id: str = Form(...),
    provider: str = Form(...),
    language: str = Form(...),
    options: str = Form("{}"),
    ref_audio: UploadFile | None = File(None),
):
    opts = json.loads(options)

    if ref_audio is not None:
        lang_dir = VOICES_DIR / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        # namespaced by provider too — the same voice_id can have a different ref
        # wav per provider (e.g. omnivoice clone vs a different provider's clone)
        dest = lang_dir / f"{voice_id}-{provider}.wav"
        dest.write_bytes(ref_audio.file.read())
        # stored relative to the repo root so voices.json stays portable across machines
        opts["ref_audio"] = dest.relative_to(REPO_ROOT).as_posix()

    return registry.put(voice_id, provider, {"language": language, "options": opts})
