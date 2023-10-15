from fastapi import FastAPI, UploadFile, File, Depends
from starlette.responses import JSONResponse
from datetime import datetime
from pydantic import BaseModel
from enum import Enum
from object_detection import get_net
from object_detection import detect

import torch
from PIL import Image
import io
from ocr import ocr, get_texts, find_date_type, find_expiration_date
from chat import propose_dish
import os

app = FastAPI()

net = get_net()
with open('config/openai_key.txt', 'r', encoding='utf-8') as file:
    os.environ["OPENAI_API_KEY"] = file.read()

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
    objects = detect(image, net)
    texts = ocr(image)

    object_num = len(objects)
    #バウンディングボックスがない場合は、画像全体を返す
    if object_num == 0:
        objects = [{
            "box":[0, 0, image.size[0], image.size[1]],
            "label":0
        }]

    data_list = []
    #バウンディングボックスごとに処理
    for obj in objects:
        box = obj["box"]
        label = obj["label"]

        #消費期限or賞味期限を抽出
        if object_num == 1: #バウンディングボックスが１つの場合は、画像全体から文字を探す
            ocr_box = [0, 0, image.size[0], image.size[1]]
        else:
            ocr_box = box
        text = get_texts(ocr_box, texts)
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

from pydantic import BaseModel, Field, validator
from typing import List
from datetime import datetime

class Ingredient(BaseModel):
    食材: str
    期限種類: str
    期限: str

class ChatRequest(BaseModel):
    食材リスト : List[Ingredient]
    目的: str

@app.post("/propose_dish/")
async def propose(query: ChatRequest):
    """提案された食材から料理を提案する"""
    ingredients = [dict(ing) for ing in query.食材リスト]

    try:
        proposed_dish = propose_dish(
            dish_num="5", 
            ingredients=ingredients, 
            today=datetime.now().date().strftime("%Y-%m-%d"), 
            condition=query.目的
        )
        return proposed_dish

    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})