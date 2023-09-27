from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
from datetime import datetime
from pydantic import BaseModel
from enum import Enum
from .neural_net import Net, detect
import torch
from PIL import Image
import io

app = FastAPI()

STATE_PATH = './model.pt'
net = Net(num_classes=1)
net.load_state_dict(torch.load(STATE_PATH))
net.eval()

LABEL_NAMES = ["background","牛乳"]

class TypeEnum(str, Enum):
    best_before = "賞味期限"
    expiration_date = "消費期限"

class Coordinate(BaseModel):
    """
    物体検出されたアイテムの座標情報を示すクラス
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float

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
    image = Image.open(io.BytesIO(contents))
    objects = detect(image, net)
    print(objects)

    data_list = []
    for obj in objects:
        box = obj["box"]
        label = obj["label"]

        data_list.append(ImageData(
            name=LABEL_NAMES[label],
            type=TypeEnum.best_before,
            date=datetime.now().strftime("%Y-%m-%d"),
            coordinate=Coordinate(xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3])
        ))
    
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
