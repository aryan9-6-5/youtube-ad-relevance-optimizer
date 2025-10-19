# src/pipelines/main.py
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
async def root():
    return {"message": "YouTube Ad Relevance Optimizer"}