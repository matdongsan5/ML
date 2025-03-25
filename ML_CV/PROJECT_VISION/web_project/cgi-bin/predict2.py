import cgi, codecs, sys
import os
import joblib
import numpy as np
from PIL import Image, ImageFile 
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
import base64
import json
import cv2


def resize_image(image, width=None, height=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR):
    """
    cv2.resize()를 활용하여 이미지를 크기 조정하는 함수

    Parameters:
        image (numpy.ndarray): 원본 이미지
        width (int, optional): 새로운 이미지의 너비 (dsize와 함께 사용)
        height (int, optional): 새로운 이미지의 높이 (dsize와 함께 사용)
        fx (float, optional): 가로 크기 비율 (dsize 대신 사용 가능)
        fy (float, optional): 세로 크기 비율 (dsize 대신 사용 가능)
        interpolation (int, optional): 보간법 (기본값 cv2.INTER_LINEAR)

    Returns:
        numpy.ndarray: 크기 조정된 이미지
    """
    if width is not None and height is not None:
        dsize = (width, height)
    else:
        dsize = None  # fx, fy가 사용됨

    resized = cv2.resize(image, dsize=dsize, fx=fx, fy=fy, interpolation=interpolation)
    return resized
def rotate_image(image, angle, center=None, scale=1.0):
    """
    이미지를 특정 각도로 회전하는 함수

    Parameters:
        image (numpy.ndarray): 원본 이미지
        angle (float): 회전 각도 (반시계 방향)
        center (tuple, optional): 회전 중심 좌표 (기본값: 이미지 중앙)
        scale (float, optional): 크기 비율 (기본값: 1.0)

    Returns:
        numpy.ndarray: 회전된 이미지
    """
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)  # 중심점을 이미지 중앙으로 설정

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
def shear_image(image, shear_x=0, shear_y=0):
    """
    이미지를 비트는(어파인 변환) 함수

    Parameters:
        image (numpy.ndarray): 원본 이미지
        shear_x (float, optional): x축 방향 기울기 정도 (기본값: 0)
        shear_y (float, optional): y축 방향 기울기 정도 (기본값: 0)

    Returns:
        numpy.ndarray: 비틀어진 이미지
    """
    (h, w) = image.shape[:2]
    
    M = np.float32([[1, shear_x, 0],
                    [shear_y, 1, 0]])
    
    sheared = cv2.warpAffine(image, M, (w, h))
    return sheared
def flip_image(image, flip_code):
    """
    이미지를 뒤집는 함수

    Parameters:
        image (numpy.ndarray): 원본 이미지
        flip_code (int): 0 (상하 뒤집기), 1 (좌우 뒤집기), -1 (상하+좌우 뒤집기)

    Returns:
        numpy.ndarray: 뒤집힌 이미지
    """
    flipped = cv2.flip(image, flip_code)
    return flipped

def remove_background(image):
    """
    rembg를 이용하여 이미지의 배경을 제거하는 함수

    Parameters:
        image (numpy.ndarray): 원본 이미지 (BGR 형식)

    Returns:
        numpy.ndarray: 배경이 제거된 이미지 (RGBA 형식)
    """
    # OpenCV의 이미지를 PIL 이미지로 변환
    import PIL.Image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 이미지는 BGR이므로 변환 필요
    pil_image = PIL.Image.fromarray(image_rgb)

    # 배경 제거 수행
    output_pil = remove(pil_image)

    # 다시 OpenCV 형식으로 변환 (RGBA 형식 유지)
    output = np.array(output_pil)
    
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGBA2BGR)

    return output_bgr
def blur_image(image, ksize=5):
    """
    이미지를 블러 처리하는 함수

    Parameters:
        image (numpy.ndarray): 원본 이미지
        ksize (int, optional): 커널 크기 (기본값: 5, 홀수만 가능)

    Returns:
        numpy.ndarray: 블러 처리된 이미지
    """
    if ksize % 2 == 0:
        raise ValueError("ksize는 홀수여야 합니다.")
    
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred


""" 
목표:
모델 로드, 이미지 받아서 모델에 넣고 예측값 반환.
"""

model = joblib.load(r'..\_model\rf_model_b.joblib')
def show_form(img, msg=""):
    print("Content-Type: text/html; charset=utf-8")
    print(f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <title>이미지 업로드</title>
        </head>
        <body>
            <form action="./predict.py" method="post" enctype="multipart/form-data">
                <input type="file" name="img">
                <input type="submit" value="업로드">
            </form>
            <p>{msg}</p>
            {"<img src='uploads/" + os.path.basename(file_path) + "' width='200'>" if img else ""}
        </body>
        </html>
    """)
    
    
def img2array(img):
    # img 받아서 1차원 만들고. 
    # 전처리 다하고
    # 
    img_arr = np.array(img)
    img_resized  = resize_image(img_arr, (70,70))
        
    img_resized.flatten()
    img_pre = model.predict(img_resized)
    
    return img_pre

def pre2img(img_pre):
    print(img_pre)
    return img_pre



# -------------------------------------------------------
# 기능 구현
# -------------------------------------------------------
# (1) WEB 인코딩 설정 -------------------------------------
sys.stdout=codecs.getwriter('utf-8')(sys.stdout.detach())
# 📌 폼 데이터 가져오기
# 📌 손상된 이미지도 강제로 로드 가능하게 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True  

# 📌 CGI 폼 데이터 가져오기
form = cgi.FieldStorage()
UPLOAD_DIR = "C:/Users/kdt/OneDrive/KDT7ML/ML_CV/PROJECT_VISION/web_project/uploads"
DEFAULT_IMAGE = "default.jpg"  # 기본 이미지 파일명
os.makedirs(UPLOAD_DIR, exist_ok=True)  # 폴더가 없으면 생성

image = None  # 🔹 이미지 변수를 미리 선언
file_path = os.path.join(UPLOAD_DIR, DEFAULT_IMAGE)  # 기본 이미지 설정
pre_ = "이미지를 업로드하세요."

if "img" in form:
    file_item = form["img"]

    if file_item.filename:
        file_path = os.path.join(UPLOAD_DIR, os.path.basename(file_item.filename))

        # 📌 파일 저장
        with open(file_path, "wb") as f:
            while chunk := file_item.file.read(1024):
                f.write(chunk)

        # 📌 파일 크기 확인
        if os.path.getsize(file_path) == 0:
            file_path = os.path.join(UPLOAD_DIR, DEFAULT_IMAGE)  # 파일이 비어 있으면 기본 이미지 사용

        try:
            # 📌 이미지 열기
            image = Image.open(file_path).convert("RGB")
        except OSError:
            file_path = os.path.join(UPLOAD_DIR, DEFAULT_IMAGE)  # 손상된 경우 기본 이미지 사용
            image = Image.open(file_path).convert("RGB")

if image:  # 🔹 사용자가 이미지를 업로드한 경우에만 모델 실행
    image = image.resize((70, 70))  # 모델 입력 크기 맞추기
    image_array = np.array(image) / 255.0  # 정규화
    
    # 📌 모델 예측
    prediction = 'x' #model.predict([image_array.flatten().reshape(1, -1)])
    pre_ = f"예측 결과: {prediction[0]}"


show_form(image, pre_)