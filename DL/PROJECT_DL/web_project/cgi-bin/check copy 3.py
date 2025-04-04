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

UPLOAD_DIR = "./upload"  # 업로드 폴더 경로

# # 폴더 내 파일 삭제 함수
# # # CGI 스크립트 실행 시 업로드 폴더 정리
# # 다음 이미지를 위한 upload 비우기
def clear_upload_folder():
    if os.path.exists(UPLOAD_DIR):
        for file in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # 파일 삭제
            except Exception as e:
                print(f"파일 삭제 오류: {file_path} - {e}")

# 📌 버튼이 눌렸는지 확인
form = cgi.FieldStorage()
if "clear_upload" in form:
    clear_upload_folder()  # 폴더 정리 실행

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
MODEL_DIR = './models/'
WEIGHTS_FILE = MODEL_DIR+'epoch13_0.905_3rd.pt'
from wfCNN3 import wfCNN
model1=wfCNN()
states=torch.load(WEIGHTS_FILE, weights_only=True)
model1.load_state_dict(states)

model2=wfCNN()


from fire_classification_oh import FDetectionCNN
WEIGHTS_FILE3 = MODEL_DIR+'Fire_DETECT_epoch7_0.808_oh.pt'
model3=FDetectionCNN()
states3=torch.load(WEIGHTS_FILE3, weights_only=True)
model1.load_state_dict(states3)

model4=wfCNN()
    

def handle_upload():
    form = cgi.FieldStorage()
    upload_message = ""
    file_path = ""
    filename = ""

    if "img" in form and form["img"].filename:
        # ✅ 직접 업로드된 경우
        fileitem = form["img"]
        filename = os.path.basename(fileitem.filename)
        file_path = os.path.join(UPLOAD_DIR, filename)

        try:
            with open(file_path, 'wb') as f:
                f.write(fileitem.file.read())
            upload_message = f"파일 '{filename}'이(가) 성공적으로 업로드되었습니다."
        except Exception as e:
            upload_message = f"파일 업로드 중 오류 발생: {str(e)}"
    
    elif "img" in form:
        # ✅ 쿼리 문자열로 전달된 경우 (자동 스크린샷)
        filename = form.getvalue("img")
        file_path = os.path.join(UPLOAD_DIR, filename)
        upload_message = f"CCTV 이미지 '{filename}' 분석 결과입니다."
    
    else:
        upload_message = "업로드할 파일을 선택해주세요."
        return upload_message, "", []

    pre_ = pre_Img(filename)
    return upload_message, file_path, pre_


def pred(img, model):
        output = model(img)  # (1, 1)
        predicted = (output.sigmoid() > 0.5).long().item()  # 0 또는 1 변환
        return predicted


def pre_Img(filename):
    if filename == "":
        return ""

    preprocessing1 = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    preprocessing3 = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    

    IMG_PATH = f"../upload/{filename}"
    
    try:
        img = Image.open(IMG_PATH).convert("RGB")

        img1 = preprocessing1(img)
        img1 = img1.unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)
        
        img3 = preprocessing3(img)
        img3 = img3.unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)
        
    except Exception as e:
        upload_message = f"파일 '{filename}' 이미지 로드 실패: {str(e)}."
        return ""

    pred1 = pred(img1, model1)
    pred2 = pred(img1, model2)
    pred3 = pred(img3, model3)
    pred4 = pred(img1, model4)

    return (pred1, pred2, pred3, pred4)


# # 업로드 처리
upload_message, file_path, pre_ = handle_upload()
# pre_path = '../origin/'+f"{pre_[0]:04d}01.png"
# if pre_:
#     pre_path = '../origin/'+f"0{pre_[0]:05d}.png"
# else: 
#     pre_path = ""
# 이미지 표시 HTML
img_html = f"<img src='.{file_path}' width='100%' alt='업로드된 이미지'>" if file_path else "이미지가 없습니다."
if not pre_:
    pre_ = [0,0,0,0]
# HTML에서 사용할 원(circle) 변환
circles = "".join(
    f"<span style='font-size: 24px; color: {'green' if p else 'red'};'> {'<td>🟢</td>' if p else '<td>🔴</td>'} </span>"
    for p in pre_
)
pre_html = f"{circles}"

# HTML 출력
print()

print(f"""

<!DOCTYPE html>
<html lang="ko">
<head>

    <meta charset="utf-8">
    <title>이미지 업로드</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; 
               display: gird;
            justify-content: center;  /* 가로 중앙 정렬 */
            align-items: center;
             width: 800px;
             margin-left:430px;
             background-image: url(./image.jpg);
            background-repeat: no-repeat;
               }}
            
        
        .upload-form {{ border: 1px solid #ddd; border-radius: 5px; 
                        justify-content: center;
                        padding: 10px; 
                        margin-left: 10px;
                        margin-top: 10px;
                        width: 100%;
        
        }}
        .message {{ position: relative; 
                    justify-content: center;
                    width: 100%; 
                    padding: 10px; 
                    margin-left: 10px;
                    margin-top: 10px;
                    border-radius: 5px; }}
        .success {{ background-color: #d4edda; color: #155724; }}
        .container{{ display: flex;
                    
                    justify-content: center;
                    width: 100%;
                    }}
                    
        .subcontainer{{ display: flex;
                        
                    position: relative; 
                    width:100%;
                    }}
        
        
        .image-container1 {{
                            align='center';
                            font-size:25px;
                            position: relative; 
                            width: 500px; 
                            height: relative;
                            margin-top: 20px;
                            margin-right:10px;
                            border: 5px solid green; 
                            ; }}
        .image-container2 {{ 
                            align='center';
                            font-size:25px;
                            position: relative; 
                            width: 90%; 
                            margin-top: 20px; 
                            margin-left:30px;
                            
                            ; }}
         p {{
             font-color: white;
         }}
         th, td {{
        border: 2px solid white;
        padding: 8px;
        text-align: center;
        }}
         
                            
    </style>
</head>
<body>
    <h1 p align='center'>산불 사진 분류</h1>
    <div class="upload-form">
        <form action="./check.py" method="post" enctype="multipart/form-data">
            <input type="file" name="img" accept="image/*">
            <input type="submit" value="분석 시작">
        </form>
                <!-- 📷 CCTV 스크린샷 버튼 -->
        <form action="./caputre_cctv.py" method="post">
            <input type="submit" value="📸 CCTV 촬영">
        </form>
        <!-- 📂 업로드 폴더 정리 버튼 -->
        <form method="post">
            <input type="hidden" name="clear_upload" value="1">
            <input type="submit" value="🗑 업로드 폴더 비우기">
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
            <table width="100%" style="color: white; background-color: black;">
            <tr>
                <td>sf</td> <td>s</td> <td>f</td> <td>nf</td>
            </tr>
            <tr>
                <td>{pre_[0]}</td> <td>{pre_[1]}</td> <td>{pre_[2]}</td> <td>{pre_[3]}</td>
            </tr>
               
               <tr>
               {circles}
               </tr>
               </table>
               <table width=100% style="color: white;">
               <tr style="color: black; background-color: #F08080;">
                <td>sf : 연기 + 불꽃 </td>
               </tr>
              <tr style="color: black; background-color: #F08080;">
               <td> f  : 불꽃 </td>
               </tr>
               <tr style="color: white; background-color: #6B8E23;">
               <td> s  : 연기 </td>
               </tr>
               <tr style="color: white; background-color: #6B8E23;">
               <td> nf : 화재 아님 </td>
               </tr>
               </table>
            </div>
          
        </div>
    </div>
</body>
</html>
""")



