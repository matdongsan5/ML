#!/usr/bin/env python3

import cgi, os.path
from pydoc import html

import joblib, sys, codecs
import cgitb
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import io
## 모듈 로딩
import torch                                        ## Tensor 및 기본 함수들 관련 모듈
import torch.nn as nn                               ## 인공신경망 관련 모듈
import torch.nn.functional as F # type: ignore
import torch.optim as optim 
from   torch.optim.lr_scheduler import ReduceLROnPlateau
from   torchmetrics.classification import *

from torchvision.datasets import FashionMNIST       ## 비젼관련 내장 데이터셋 모듈
from torch.utils.data import DataLoader             ## Pytorch의 데이터셋 관련 모듈
from torchinfo import summary                       ## 모델 구조 및 정보 확인 모듈

import torchvision.transforms as transforms         ## 비젼관련 이미지 증강/변환 관련 모듈

import matplotlib.pyplot as plt                     ## 이미지 시각화 

from torchvision.datasets import ImageFolder            ## 이미지용 데이터셋 생성 모듈\n",
from torch.utils.data import DataLoader                 ## 데이터로더\n",
from torchvision.transforms import transforms           ## 이미지 전처리 및 증강 모듈


# stdout 인코딩을 utf-8로 설정
## 한글 출력 부분.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

# CGI 디버깅 활성화
cgitb.enable()

# 업로드된 파일을 저장할 디렉토리 설정
UPLOAD_DIR = "./upload"  # 업로드 디렉토리 경로
                        #./upload

# 업로드 디렉토리가 없으면 생성
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

#헤더 출력. 한글출력부분
print("Content-Type: text/html; charset=utf-8\n")
## 방법-2) 저장된 층별 가중치와바이어스 로딩 
MODEL_DIR = '../models/'
WEIGHTS_FILE = MODEL_DIR+'fashion_weights_epoch61_0.949.pt'
from wfCNN import wfCNN
model1=wfCNN()
states=torch.load(WEIGHTS_FILE, weights_only=True)
model1.load_state_dict(states)

model2=wfCNN()
model3=wfCNN()
model4=wfCNN()
    

def handle_upload():
    form = cgi.FieldStorage()
    upload_message = ""
    file_path = ""
    filename = ""
    # 이미지 파일 처리
    if "img" in form and form["img"].filename:
        fileitem = form["img"]
        
        # 파일명 가져오기
        filename = os.path.basename(fileitem.filename)
        
        # 파일 저장 경로 설정
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # 파일 저장
        try:
            # image_data = fileitem.file.read()
            # image_stream = io.BytesIO(image_data)
            # img = Image.open(image_stream)
            # img.save(file_path)
            with open(file_path, 'wb') as f:
                f.write(fileitem.file.read())
            upload_message = f"파일 '{filename}'이(가) 성공적으로 업로드되었습니다."
        except Exception as e:
            upload_message = f"파일 업로드 중 오류 발생: {str(e)}"
    else:
        upload_message = "업로드할 파일을 선택해주세요."
    pre_ = pre_Img(filename)
    
    return upload_message, file_path, pre_


def pred(img, model):
        output = model(img)  # (1, 1)
        predicted = (output.sigmoid() > 0.5).long().item()  # 0 또는 1 변환
        return predicted


def pre_Img(filename):
    if filename == "":
        return ""

    preprocessing = transforms.Compose([
        transforms.Resize((50, 50), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    IMG_PATH = f"./upload/{filename}"
    
    try:
        img = Image.open(IMG_PATH).convert("RGB")
        img = preprocessing(img)
        img = img.unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)
    except Exception as e:
        print(f"이미지 로드 실패: {str(e)}")
        return ""

    pred1 = pred(img, model1)
    pred2 = pred(img, model2)
    pred3 = pred(img, model3)
    pred4 = pred(img, model4)

    return (pred1, pred2, pred3, pred4)


# # 업로드 처리
upload_message, file_path, pre_ = handle_upload()
# pre_path = '../origin/'+f"{pre_[0]:04d}01.png"
if pre_:
    pre_path = '../origin/'+f"0{pre_[0]:05d}.png"
else: 
    pre_path = ""
# 이미지 표시 HTML
img_html = f"<img src='.{file_path}' width='100%' alt='업로드된 이미지'>" if file_path else "이미지가 없습니다."
pre_html = f"<img src='{pre_path}' width='100%' alt='업로드된 이미지'>" if pre_path else "이미지가 없습니다."

# HTML 출력
print()

print(f"""

<!DOCTYPE html>
<html lang="ko">
<head>

    <meta charset="utf-8">
    <title>이미지 업로드</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .upload-form {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .message {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .success {{ background-color: #d4edda; color: #155724; }}
        .container{{ 
                    width: 100%;
                    }}
                    
        .subcontainer{{ display: flex;
                    width: 45%;
                    }}
        
        
        .image-container1 {{
                            position: relative; 
                            z-index: 2; 
                            width: 45%; 
                            margin-top: 20px; 
                            
                            border: 1px solid #eee; 
                            padding: 10px; }}
        .image-container2 {{ 
                            position: relative; 
                            z-index: 1;
                            width: 30%; 
                            margin-top: 20px; 
                            
                            border: 1px solid #eee; 
                            padding: 10px; }}
                            
    </style>
</head>
<body>
    <h1>띠부띠부 씰 분류</h1>
    
    <div class="upload-form">
        <form action="./predict.py" method="post" enctype="multipart/form-data">
            <input type="file" name="img" accept="image/*">
            <input type="submit" value="분석 시작">
        </form>
    </div>
    
    <div class="message success">
        {upload_message}
    </div>
    <div class='container'>
        <div class = 'sub_container'>
            <div class="image-container1">
                {img_html}
            </div>
        </div>
        <div class = 'sub_container'>
            <div class="image-container2">
                {pre_}{pre_html}
            </div>
            <div class="image-container2">
                {pre_html}
            </div>
        </div>
    </div>
</body>
</html>
""")