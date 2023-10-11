import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from torchvision import transforms

NMS_THRESHOLD = 0.05
SELECT_THRESHOLD = 0.7

def get_net():
    STATE_PATH = './model.pt'
    net = Net(num_classes=1)
    net.load_state_dict(torch.load(STATE_PATH))
    net.eval()

    return net

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        #全パラメータの重みを更新しない
        for param in self.feature.parameters():
            param.requires_grad = False

        #出力層を付け替え
        in_features= self.feature.roi_heads.box_predictor.cls_score.in_features
        self.feature.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

    def forward(self, images):
        h = self.feature(images, None)
        return h

def apply_nms(input_dict, iou_threshold=0.5):
    # バウンディングボックスとスコアを抽出
    boxes = input_dict['boxes']
    scores = input_dict['scores']

    # NMSを適用
    keep_indices = nms(boxes, scores, iou_threshold)

    # 結果の辞書を作成
    result = {
        'boxes': boxes[keep_indices],
        'scores': scores[keep_indices],
        'labels': input_dict['labels'][keep_indices]
    }
    
    return result

def select_bbox(pred, threshold):
    selected_pred = []
    for box,score,label in zip(pred["boxes"], pred["scores"], pred["labels"]):
        if score > threshold:
            selected_pred.append({
                "box": box.int().tolist(),
                "score": float(score.item()),
                "label": int(label.item())
            })
    return selected_pred

def detect(image, net):
    #入力データの変換の定義
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    x = transform(image)
    y = net(x.unsqueeze(0))[0]
    y_nms = apply_nms(y, NMS_THRESHOLD)
    selected_box = select_bbox(y_nms, SELECT_TH)
    return selected_box



from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import numpy as np

def get_net_detectron2():
    #パラメータファイル設定
    cfg = get_cfg()
    cfg.merge_from_file("config.yaml")
    cfg.MODEL.WEIGHTS = "model.pth"
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    predictor = DefaultPredictor(cfg)
    return predictor

def detect_detectron2(image, predictor):
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
