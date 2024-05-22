from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
import io
import numpy as np

# FastAPI 인스턴스 생성
app = FastAPI()

# 모델 로드
model_path = "aram_model6.pt"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

def transform_image(image_bytes):
    """이미지 바이트를 받아 모델 입력 형태로 변환"""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))  # 모델 입력 크기에 맞게 조정
    image = np.array(image).astype(np.float32) / 255.0  # 정규화
    if len(image.shape) == 2:  # 흑백 이미지일 경우 채널 추가
        image = np.expand_dims(image, axis=2)
    if image.shape[2] == 1:  # 흑백 이미지를 3채널로 변환
        image = np.repeat(image, 3, axis=2)
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    return image

# 예측 엔드포인트
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_data = transform_image(image_bytes)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        # 모델 예측
        with torch.no_grad():
            output = model(input_tensor)

        # 결과 반환
        return {"prediction": output.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 테스트 엔드포인트
@app.get("/")
def read_root():
    return {"Hello": "World"}
