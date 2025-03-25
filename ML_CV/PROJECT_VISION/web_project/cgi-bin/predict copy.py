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


def show_form(img, pre_img=""):
    print("Content-Type: text/html; charset=utf-8")
    print("""
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
            <div id='div1' name='div1'>{0}</div>
            <div id='div2' name='div2'>{1}</div>
        </body>
        </html>
    """.format(img, pre_img))
    
    
def img2array(img):
    # img 받아서 1차원 만들고. 
    # 전처리 다하고
    img_arr = np.array(img)
    img_resized  = resize_image(img_arr, (70,70))
        
    img_resized.flatten()
    img_pre = model.predict(img_resized)
    
    return img_pre

def pre2img(img_pre):
    print(img_pre)
    return img_pre

image = "업로드된 이미지가 여기 표시됩니다"
pre_img = "예측 결과가 여기 표시됩니다"


sys.stdout=codecs.getwriter('utf-8')(sys.stdout.detach())
ImageFile.LOAD_TRUNCATED_IMAGES = True  
model = joblib.load(r'..\_model\rf_model_b.joblib')
form = cgi.FieldStorage()
image_data = form.getvalue("img", default="1")
if image_data:
    image = image_data
    pre_img = img2array(image)
else:
    image_data = 'None'

show_form(image, pre_img)