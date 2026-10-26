from fastapi import FastAPI

app = FastAPI(title="Project 130 CLV API")


@app.get("/health")
def health_check():
    return {"status": "ok"}