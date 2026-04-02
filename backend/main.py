import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import engine, Base
import models  # noqa: F401 — ensures all tables are registered before create_all

Base.metadata.create_all(bind=engine)

app = FastAPI(title="大模型金融分析 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers import portfolio, trades, signals, scan, dashboard

app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(trades.router,    prefix="/api/trades",    tags=["trades"])
app.include_router(signals.router,   prefix="/api/signals",   tags=["signals"])
app.include_router(scan.router,      prefix="/api/scan",      tags=["scan"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
