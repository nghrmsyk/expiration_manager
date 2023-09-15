from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
from datetime import datetime
from pydantic import BaseModel
from enum import Enum

app = FastAPI()

class TypeEnum(str, Enum):
    best_before = "賞味期限"
    expiration_date = "消費期限"

class Coordinate(BaseModel):
    """
    物体検出されたアイテムの座標情報を示すクラス
    """
    cx: float
    cy: float
    w: float
    h: float

class ImageData(BaseModel):
    """
    物体検出されたアイテムの詳細情報を示すクラス
    """
    name: str
    type: TypeEnum
    date: str
    coordinate: Coordinate

def process_image(contents: bytes) -> list[ImageData]:
    """アップロードされた画像を処理し、食品とその消費期限を検出する"""
    
    data_list = [
        ImageData(
            name="サンプル名1",
            type=TypeEnum.best_before,
            date=datetime.now().strftime("%Y-%m-%d"),
            coordinate=Coordinate(cx=0.5, cy=0.5, w=1, h=1)
        ),
        ImageData(
            name="サンプル名2",
            type=TypeEnum.expiration_date,
            date=datetime.now().strftime("%Y-%m-%d"),
            coordinate=Coordinate(cx=0.5, cy=0.5, w=0.5, h=0.5)
        )
    ]
    
    return data_list

@app.post("/food-expiration/")
async def detect_expiration(file: UploadFile = File(...)):
    """アップロードされた食品の画像から消費期限を検出する"""
    
    try:
        contents = await file.read()
        data_list = process_image(contents)
        return {"data": data_list}
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
