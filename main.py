import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
from routers import ocr as ocr_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(ocr_router.router)


@app.get("/")
def root():
    return {"status": "ok", "services": ["ocr_api"]}


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}


if os.getenv("ENABLE_OCR_ANNOTATOR", "0") == "1":
    from routers import ocr_tools as ocr_tools_router

    app.include_router(ocr_tools_router.router)


if __name__ == "__main__":
    reload_on = os.getenv("DEV", "0") == "1"
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        reload=reload_on,
        reload_dirs=["routers", "services", "models", "storage"]
    )
