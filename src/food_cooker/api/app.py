from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from food_cooker.api.routes import chat, auth, vision


def create_app() -> FastAPI:
    app = FastAPI(title="FoodCooker API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(vision.router, prefix="/api")
    return app
