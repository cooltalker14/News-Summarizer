from fastapi import FastAPI, HTTPException
from utils import generate_report
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/report")
async def get_report(company: str):
    report, audio_file = generate_report(company)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])
    return {"report": report, "audio": audio_file}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
