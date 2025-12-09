from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.image_routes import router as image_router
from app.routes.video_routes import router as video_router
from app.routes.realtime_routes import router as realtime_router

app = FastAPI(title="Fire & Smoke Detection API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.1.2:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(image_router, prefix="/api/image")
app.include_router(video_router, prefix="/api/video")
app.include_router(realtime_router, prefix="/api/realtime")

@app.get("/")
def root():
    return {"message": "Fire & Smoke Detection API Running"}
