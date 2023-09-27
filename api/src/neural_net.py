import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from torchvision import transforms

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.feature = fasterrcnn_resnet50_fpn_v2(pretrained=True)
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
    y_nms = apply_nms(y, 0.05)
    selected_box = select_bbox(y_nms,0.5)
    return selected_box
