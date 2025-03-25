#!/usr/bin/env python3

import cgi, os.path
from pydoc import html

import joblib, sys, codecs
import cgitb
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import io

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

def pre_Img(filename):
    
    if filename =="":
     return ""
    model = joblib.load('./_model/rf_model_sbs.joblib')
    
    if ('./upload/'+filename).endswith('png'):
        pillow_image = Image.open('./upload/'+filename)
        numpy_image = np.array(pillow_image)
        examImg = cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2RGB)
        a = examImg
        # a = cv2.cvtColor(examImg, cv2.COLOR_BGR2RGB)
        # a = cv2.imread('../upload/'+filename, cv2.IMREAD_UNCHANGED)
        # a = cv2.cvtColor(a, cv2.COLOR_BGRA2BGR) 
        
    else:    
        a = cv2.imread('./upload/'+filename, cv2.IMREAD_COLOR)
        a = cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2RGB)
    b = resize_image(a, 70, 70)
    c = b.flatten()
    resultDF = pd.DataFrame(columns=list(range(1,14701)))
    resultDF.loc['0'] = c
    resultDF.columns.astype(str)
    pre_ = model.predict(resultDF/255.)
    # pre_ = model.predict(resultDF)
    # pre_ = f"{c:04d}01.png"
    return pre_ 

# # 업로드 처리
upload_message, file_path, pre_ = handle_upload()
# pre_path = '../origin/'+f"{pre_[0]:04d}01.png"
if pre_:
    pre_path = '../origin/'+f"0{pre_[0]:03d}01.png"
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
        body {{ font-family: Arial, sans-serif; margin: 20px; 
               display: gird;
            justify-content: center;  /* 가로 중앙 정렬 */
            align-items: center;
             width: 800px;
             margin-left:430px;
               }}
        h1 {{ p align: center; }}       
        
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
                    width: 45%;
                    }}
        
        
        .image-container1 {{
                            align='center';
                            font-size:25px;
                            position: relative; 
                            width: 100%; 
                            margin-top: 20px;
                            margin-right:30px;
                            border: 5px solid #FFD700; 
                            ; }}
        .image-container2 {{ 
                            align='center';
                            font-size:25px;
                            position: relative; 
                            width: 100%; 
                            margin-top: 20px; 
                            margin-left:30px;
                            
                            border: 5px solid #006400; 
                            ; }}
                            
    </style>
</head>
<body>
    <h1 p align='center'>띠부띠부 씰 분류</h1>
    
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
             <p p align='center'> 띠부띠부씰 번호 &nbsp[&nbsp{file_path[-9:-6]}.{file_path[-6:-3]}&nbsp] </p>
            </div>
            <div class="image-container1">
                {img_html}
            </div>
        </div>
        <div class = 'sub_container'>
            <div class="image-container2">
              <p p align='center'> 예상 도감번호 &nbsp[&nbsp{pre_[0]}&nbsp]  </p>  
            </div>
            <div class="image-container2">
                {pre_html}
            </div>
        </div>
    </div>
</body>
</html>
""")