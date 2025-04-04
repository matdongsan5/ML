from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import time
import os

def capture_cctv_frame(url, save_path='cctv_frame.png'):
    # 셀레니움 설정
    options = Options()
    options.add_argument('--headless')  # 창 안 띄움
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1280x720')

    # 크롬 드라이버 경로 지정
    driver = webdriver.Chrome(options=options)

    try:
        # URL 열기
        driver.get(url)

        time.sleep(5)  # 스트리밍 로딩 시간 확보 (필요시 더 늘려도 됨)

        # 전체 페이지 스크린샷
        screenshot_path = 'full_screenshot.png'
        driver.save_screenshot(screenshot_path)

        # 이미지 열고 원하는 영역만 자르기 (좌표 수동 조정 필요)
        image = Image.open(screenshot_path)
        
        # CCTV 영상이 표시되는 위치 (화면 기준 좌표)
        crop_box = (100, 150, 1180, 700)  # (left, top, right, bottom)
        cropped = image.crop(crop_box)
        cropped.save(save_path)

        print(f'CCTV 영상 캡처 완료: {save_path}')
    finally:
        driver.quit()
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)

# ▶ 실행 예시
url = "https://www.utic.go.kr/jsp/map/cctvStream.jsp?cctvid=E902565&cctvname=%255B%25EA%25B5%25AD%25EB%258F%258435%255D%25EC%25B2%25AD%25EC%2586%25A1%25EC%259B%2594%25EC%25A0%2595%25EC%2582%25BC%25EA%25B1%25B0%25EB%25A6%25AC&kind=Z3&cctvip=undefined&cctvch=undefined&id=74298/Wb/MvrANESnU6QFDHg7J75s0gOKEiYKSAZjicY6jUXsv1%2BIVspFrtjUowcZFMzcKbiDxsaNvBeHr9ogUGDTZyANINzCJUGN1avKMkRFm5SM=&cctvpasswd=undefined&cctvport=undefined&minX=128.80663607346315&minY=36.11516050197272&maxX=129.0561574654073&maxY=36.23389039194848"
capture_cctv_frame(url, 'my_cctv_shot.png')