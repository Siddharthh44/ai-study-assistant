from fastapi import FastAPI

app=FastAPI()

@app.get("/")
def read_root():
    return {"message":"AI Study Assistant Backend Running"}

@app.get("/health")
def health_check():
    return {"status":"ok"}
