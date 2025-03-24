#!/usr/bin/env python3
import cgi
import cgitb
# 스크립트 상단에 추가
import os
import sys 
import tempfile
from PIL import Image
import io
import base64
import json
import time

os.environ['LANG'] = 'ko_KR.UTF-8'
# CGI 오류 추적 활성화
cgitb.enable()

# 업로드 디렉토리 설정
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_image(image_path):
    """
    이미지를 처리하고 결과 이미지를 반환하는 함수
    실제 구현에서는 여기에 모델 예측 코드를 넣으면 됩니다
    """
    # 원본 이미지 로드
    img = Image.open(image_path)
    
    # 예시: 이미지에 간단한 처리 적용 (세피아 톤)
    # 실제로는 여기서 모델을 호출하여 예측 결과에 따른 이미지 생성
    sepia = []
    width, height = img.size
    
    # 세피아 톤 변환 (모델 예측 시뮬레이션)
    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))[:3]
            new_r = int(min(255, r * 0.393 + g * 0.769 + b * 0.189))
            new_g = int(min(255, r * 0.349 + g * 0.686 + b * 0.168))
            new_b = int(min(255, r * 0.272 + g * 0.534 + b * 0.131))
            sepia.append((new_r, new_g, new_b))
    
    # 결과 이미지 생성
    result_img = Image.new('RGB', (width, height))
    result_img.putdata(sepia)
    
    # 처리 결과 저장
    result_path = image_path.replace('.', '_result.')
    result_img.save(result_path)
    
    return result_path

def main():
    # HTTP 헤더 출력
    print("Content-Type: text/html")
    print()
    
    # HTML 시작 부분
    print("""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>이미지 처리 결과</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }
        
        .container {
            position: relative;
            width: 1000px;
            height: 700px;
            padding: 20px;
        }
        
        .frame1 {
            position: absolute;
            top: 0;
            left: 0;
            width: 60%;
            aspect-ratio: 1 / 1;
            border: 2px solid black;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 24px;
            background-color: white;
            z-index: 1;
            overflow: hidden;
        }
        
        .frame2 {
            position: absolute;
            top: 100px;
            right: 0;
            width: 50%;
            aspect-ratio: 1 / 1;
            background-color: #FF8A3C;
            border-radius: 25px;
            display: flex;
            justify-content: center;
            align-items: center;
            color: white;
            font-size: 24px;
            z-index: 2;
            overflow: hidden;
        }
        
        .back-btn {
            position: absolute;
            bottom: 20px;
            left: 0;
            width: 60%;
            text-align: center;
        }
        
        .btn {
            padding: 10px 20px;
            background-color: #4285F4;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            text-decoration: none;
        }
        
        img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
    </style>
</head>
<body>
    <div class="container">""")
    
    try:
        # form 데이터 파싱
        form = cgi.FieldStorage()
        
        # 이미지 파일 가져오기
        fileitem = form['image']
        
        # 이미지 파일이 실제로 업로드되었는지 확인
        if fileitem.filename:
            # 임시 파일에 저장
            file_path = os.path.join(UPLOAD_DIR, os.path.basename(fileitem.filename))
            
            # 파일 열기 및 내용 쓰기
            with open(file_path, 'wb') as f:
                f.write(fileitem.file.read())
            
            # 이미지 처리 (모델 예측)
            # 잠시 처리 중임을 보여주기 위한 대기
            time.sleep(1)
            result_path = process_image(file_path)
            
            # 원본 이미지와 결과 이미지 경로
            original_path = '/uploads/' + os.path.basename(file_path)
            result_path = '/uploads/' + os.path.basename(result_path)
            
            # 이미지 표시
            print(f"""
        <div class="frame1">
            <img src="{original_path}" alt="원본 이미지">
        </div>
        
        <div class="frame2">
            <img src="{result_path}" alt="예측 결과 이미지">
        </div>
            """)
        else:
            print("""
        <div class="frame1">
            <p>이미지가 업로드되지 않았습니다.</p>
        </div>
        
        <div class="frame2">
            <p>예측 결과가 없습니다.</p>
        </div>
            """)
    except Exception as e:
        print(f"""
        <div class="frame1">
            <p>오류가 발생했습니다: {str(e)}</p>
        </div>
        
        <div class="frame2">
            <p>처리할 수 없습니다.</p>
        </div>
        """)
    
    # 뒤로 가기 버튼
    print("""
        <div class="back-btn">
            <a href="/" class="btn">다시 시작하기</a>
        </div>
    </div>
</body>
</html>""")

if __name__ == "__main__":
    main()