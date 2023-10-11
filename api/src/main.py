from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
from datetime import datetime
from pydantic import BaseModel
from enum import Enum
from .neural_net import get_net_detectron2
from .neural_net import detect_detectron2

import torch
from PIL import Image
import io
from .ocr import ocr, get_texts, find_date_type, find_expiration_date

app = FastAPI()

net = get_net_detectron2()

LABEL_NAMES = ["","卵","牛乳","食パン"]

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
    type: str
    date: str
    coordinate: Coordinate

def process_image(contents: bytes) -> list[ImageData]:
    """アップロードされた画像を処理し、食品とその消費期限を検出する"""
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    print("物体検出スタート")
    objects = detect_detectron2(image, net)
    print(f"物体検出完了 :{objects}")
    texts = ocr(image)

    object_num = len(objects)
    #バウンディングボックスがない場合は、画像全体を返す
    if object_num == 0:
        objects = [{
            "box":[0, 0, image.size[0], image.size[1]],
            "label":0
        }]
        print(f"物体なかった: {objects}")

    data_list = []
    #バウンディングボックスごとに処理
    for obj in objects:
        box = obj["box"]
        label = obj["label"]

        #消費期限or賞味期限を抽出
        if object_num == 1: #バウンディングボックスが１つの場合は、画像全体から文字を探す
            ocr_box = [0, 0, image.size[0], image.size[1]]
            print(f"物体１つだけ: {ocr_box}")
        else:
            ocr_box = box
            print(f"物体１つ以外: {ocr_box}")
        text = get_texts(ocr_box, texts)
        print(text)
        date = find_expiration_date(text)
        date_type = find_date_type(text)

        data_list.append(ImageData(
            name=LABEL_NAMES[label],
            type=date_type,
            date=date,
            coordinate=Coordinate(xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3])
        ))
    print(data_list)
    
    return data_list

@app.post("/food-expiration/")
async def detect_expiration(file: UploadFile = File(...)):
    """アップロードされた食品の画像から消費期限を検出する"""
    
    try:
        contents = await file.read()
        #物体検出
        data_list = process_image(contents)
        return {"data": data_list}
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
