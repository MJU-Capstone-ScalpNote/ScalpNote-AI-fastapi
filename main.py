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
model_paths = {
    "model2": "checkpoint_model2.pt",
    "model3": "checkpoint_model3.pt",
    "model6": "checkpoint_model6.pt"
}

models = {}

for model_key, model_path in model_paths.items():
    model = torch.load(model_path, map_location=torch.device('cpu'))    
    model.eval()
    models[model_key] = model
# model2 = torch.load(model_path_2, map_location=torch.device('cpu'))
# model3 = torch.load(model_path_3, map_location=torch.device('cpu'))
# model6 = torch.load(model_path_6, map_location=torch.device('cpu'))

# model2.eval()
# model3.eval()
# model6.eval()

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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing the image: {e}")
        # 모델 예측
    
        # with torch.no_grad():
        #     for model_key, model in models.items():
        #         output = model(image)
        #         predictions[model_key] = output.argmax(dim=1, keepdim=True).item()
        #     output = model(input_tensor)

        # 결과 반환
        return {"prediction": output.tolist()}
    predictions = {}
    try:
        with torch.no_grad():
            for model_key, model in models.items():
                output = model(input_tensor)
                predictions[model_key] = output.argmax(dim=1, keepdim=True).item()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
    
    # Diagnosis logic
    diagnosis = []
    m2p, m3p, m6p = predictions["model2"], predictions["model3"], predictions["model6"]

    if m2p == 0 and m3p == 0 and m6p == 0:
        diagnosis.append('정상입니다.')
    if m2p != 0:
        if m2p == 1:
            diagnosis.append('피지과다 (경증)')
        elif m2p == 2:
            diagnosis.append('피지과다 (중증)')
        elif m2p == 3:
            diagnosis.append('피지과다 (중증도)')
    if m3p != 0:
        if m3p == 1:
            diagnosis.append('모낭사이홍반 (경증)')
        elif m3p == 2:
            diagnosis.append('모낭사이홍반 (중증)')
        elif m3p == 3:
            diagnosis.append('모낭사이홍반 (중증도)')
    if m2p != 0 and m3p != 0:
        if m2p == 1 and m3p == 1:
            diagnosis.append('피지과다, 모낭사이홍반 (경증)')
        elif m2p == 2 and m3p == 2:
            diagnosis.append('피지과다, 모낭사이홍반 (중증)')
        elif m2p == 3 and m3p == 3:
            diagnosis.append('피지과다, 모낭사이홍반 (중증도)')
    if m2p != 0 and m6p != 0:
        if m2p == 1 and m6p == 1:
            diagnosis.append('피지과다, 탈모 (경증)')
        elif m2p == 2 and m6p == 2:
            diagnosis.append('피지과다, 탈모 (중증)')
        elif m2p == 3 and m6p == 3:
            diagnosis.append('피지과다, 탈모 (중증도)')
    if m3p != 0 and m6p != 0:
        if m3p == 1 and m6p == 1:
            diagnosis.append('모낭사이홍반, 탈모 (경증)')
        elif m3p == 2 and m6p == 2:
            diagnosis.append('모낭사이홍반, 탈모 (중증)')
        elif m3p == 3 and m6p == 3:
            diagnosis.append('모낭사이홍반, 탈모 (중증도)')
    if m6p != 0:
        if m6p == 1:
            diagnosis.append('탈모 (경증)')
        elif m6p == 2:
            diagnosis.append('탈모 (중증)')
        elif m6p == 3:
            diagnosis.append('탈모 (중증도)')

    return {"prediction": ", ".join(diagnosis)}

# 테스트 엔드포인트
@app.get("/")
def read_root():
    return {"Hello": "World"}
