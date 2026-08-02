from fastapi import APIRouter

from .providers import router as providers_router
from .tts import router as tts_router
from .voices import router as voices_router

router = APIRouter()
router.include_router(providers_router)
router.include_router(voices_router)
router.include_router(tts_router)
