from dataclasses import asdict

from fastapi import APIRouter

from providers import PROVIDER_INFO

router = APIRouter()


@router.get("/providers")
def get_providers():
    return {name: asdict(info) for name, info in PROVIDER_INFO.items()}
