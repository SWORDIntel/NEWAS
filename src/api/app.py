from fastapi import FastAPI
from src.api.v1.dashboard import dashboard_api

app = FastAPI(
    title="NEMWAS API",
    description="Neural-Enhanced Multi-Workforce Agent System API",
    version="1.0.0"
)

app.include_router(dashboard_api.router)
