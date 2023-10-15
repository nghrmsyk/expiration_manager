import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from torchvision import transforms
import json

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import numpy as np
import config

def get_labels():
    """ラベル名を取得"""
    #ラベル読み込み
    with open(config.DETECTRON2_LABELS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
        labels = [data[str(i)] for i in range(len(data))]
        return labels

def get_net():
    #パラメータファイル設定
    cfg = get_cfg()
    cfg.merge_from_file(config.DETECTRON2_CONFIG_PATH)
    cfg.MODEL.WEIGHTS = config.DETECTRON2_MODEL_WEIGHT_PATH 
    cfg.MODEL.DEVICE = "cpu"

    #ラベル名取得
    labels = get_labels()
    cfg.MODEL.NUM_CLASSES = len(labels)
    
    predictor = DefaultPredictor(cfg)
    return predictor, labels

def detect(image, predictor):
    image = np.array(image)[:, :, ::-1]
    outputs = predictor(image)

    boxes = outputs["instances"].pred_boxes.tensor
    scores = outputs["instances"].scores
    classes = outputs["instances"].pred_classes

    selected_box = [{
        "box": box.int().tolist(),
        "score": float(score),
        "label": int(label)
    }for box, score, label in zip(boxes, scores, classes)]

    return selected_box
