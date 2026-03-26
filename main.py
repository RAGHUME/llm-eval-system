"""
FastAPI Application Entry Point
=================================

WHAT: Creates and configures the FastAPI application instance.

WHY:  This is the single entry point that ties together routes,
      templates, static files, and database initialization.

HOW:  Mounts Jinja2 templates, includes API routes, creates the
      data directory, and initializes the database on startup.

RUN:  uvicorn main:app --reload --port 8000
"""

import os
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.database import create_tables

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="LLM Evaluation & Prompt Optimization System",
    description="Evaluate, compare, and optimize LLM prompts with 7 metrics",
    version="1.0.0",
)

# Mount static files directory
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ---------------------------------------------------------------------------
# Startup event — initialize database
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize database tables on app startup."""
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    create_tables()
    logging.info("✅ Database initialized")
    logging.info("🚀 LLM Eval System is running at http://localhost:8000")

# ---------------------------------------------------------------------------
# Include API routes
# ---------------------------------------------------------------------------
from api.routes import router
app.include_router(router)
