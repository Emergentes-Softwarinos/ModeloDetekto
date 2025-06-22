from fastapi import FastAPI
from apps.recognition.urls import router as recognition_router

app = FastAPI()

# Incluir las rutas de la aplicación de reconocimiento
app.include_router(recognition_router)
