import torch
from efficientnet_pytorch import EfficientNet

# 모델 파일 경로
model_path = "aram_model6.pt"

# 모델 로드
model = torch.load(model_path, map_location=torch.device('cpu'))
print(model)