# api.py
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from main import PipeLine
import wfdb                                           # чтение .hea/.dat
import uvicorn

from config import MISTRAL_URL

config = {
        "signal_analytic": {
            "backbone": {
                "projection": {
                    "in_channels":   [12, 32, 64  ],
                    "out_channels":  [32, 64, 128 ],
                    "embedding_dim": [32, 32, 64  ]
                },
                "conv": {
                    "kernel_size": (3, 3),
                    "padding": 1,
                    "stride": 2
                },
                "pool": {
                    "kernel_size": (3, 3),
                    "padding": 1,
                    "stride": 1
                }
            },
            "out_head": {
                "type": "multyoutput",
                "params": {
                    "embedding_dim": 16,
                    "dims": [4]
                }
            }
        },
        "generation_config": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "num_return_sequences": 1
        },
        "gen_details": {
            "роль": "Ты помощник медицинского эксперта",
            "задача": """
                Ты должен генерировать промежуточные диагнозы на 
                основании диагнозов параметров экг таких как 
                RR, PR, QT, QRS, которые ты будешь получать 
                в формате текста но в виде list.
            """,
            "результат": """
                Текст с полным описанием экг пациента по 
                RR, PR, QT, QRS, дополнительными рекомендациями
                от себя, и предупрежедением о том что ты лишь 
                ИИ ассистент и пользователю стоит обратиться за 
                более точной консультацией к медработнику. 
                Все поля раздели между собой линиями.
            """
        }
    }

# инициализируем один раз при старте контейнера
pipeline = PipeLine(
    config=config,
    weights="meta/model_weights_1.pt"
)

app = FastAPI(title="ECG report generator")


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/ecg")
async def analyze_ecg(file: UploadFile = File(...)):
    # принимаем zip или одиночный WFDB-файл
    if not file.filename.endswith((".hea", ".dat", ".zip")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())

        try:
            report = pipeline(path, MISTRAL_URL=MISTRAL_URL)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(report)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, workers=1)
