import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api.routes import router
from app.core.config import get_settings, BASE_DIR
from app.core.logging import configure_logging
from app.rag.vectorstore import warm_up
from app.rag.workflow import llm, web_search_tool


configure_logging()
settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up models, Pinecone and LLM clients...")
    try:
        warm_up()
        llm()
        web_search_tool()
        logger.info("Warm-up complete")
    except Exception:
        logger.exception("Warm-up failed, components will load on first request")
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

app.include_router(router)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "app_name": settings.app_name})
