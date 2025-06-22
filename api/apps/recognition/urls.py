from fastapi import APIRouter
from .views import router as recognition_router

router = APIRouter()

router.include_router(recognition_router, prefix="/recognition", tags=["recognition"])
