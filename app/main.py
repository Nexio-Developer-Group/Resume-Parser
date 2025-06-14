from fastapi import FastAPI
from app.middleware.auth import check_api_key
from app.routes import resume

app = FastAPI()

app.middleware("http")(check_api_key)
app.include_router(resume.router)
