import torch.nn as nn
import torchvision.models as models

class SimpleCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(SimpleCNN, self).__init__()
        # ResNet50 BackBone 불러오기 (ImageNet 사전학습)
        self.backbone = models.resnet50(pretrained=pretrained)
        # 마지막 fully-connected 레이어를 이진 분류용으로 교체
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.backbone(x)