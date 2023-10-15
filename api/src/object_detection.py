import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from torchvision import transforms

NMS_THRESHOLD = 0.05
SELECT_THRESHOLD = 0.7

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import numpy as np

def get_net():
    #パラメータファイル設定
    cfg = get_cfg()
    cfg.merge_from_file("config.yaml")
    cfg.MODEL.WEIGHTS = "model.pth"
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    predictor = DefaultPredictor(cfg)
    return predictor

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
