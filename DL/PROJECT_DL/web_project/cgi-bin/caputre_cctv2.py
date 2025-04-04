

# !/usr/bin/env python3
import time
import sys, os, codecs
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
# stdout 인코딩을 utf-8로 설정
## 한글 출력 부분.
# sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

# CCTV URL
CCTV_URL = "https://www.utic.go.kr/jsp/map/cctvStream.jsp?cctvid=E902565&cctvname=%255B%25EA%25B5%25AD%25EB%258F%258435%255D%25EC%25B2%25AD%25EC%2586%25A1%25EC%259B%2594%25EC%25A0%2595%25EC%2582%25BC%25EA%25B1%25B0%25EB%25A6%25AC&kind=Z3&cctvip=undefined&cctvch=undefined&id=74298/Wb/MvrANESnU6QFDHg7J75s0gOKEiYKSAZjicY6jUXsv1%2BIVspFrtjUowcZFMzcKbiDxsaNvBeHr9ogUGDTZyANINzCJUGN1avKMkRFm5SM=&cctvpasswd=undefined&cctvport=undefined&minX=128.80663607346315&minY=36.11516050197272&maxX=129.0561574654073&maxY=36.23389039194848"
# CCTV_URL = "https://www.utic.go.kr/jsp/map/cctvStream.jsp?cctvid=E902591&cctvname=%255B%25EA%25B5%25AD%25EB%258F%258435%25ED%2598%25B8%25EC%2584%25A0%255D%25EC%2598%2581%25EC%25B2%259C%25EB%25B3%25B4%25ED%2598%2584%25EC%2582%25B0%25EB%258C%2590%25EA%25B3%25B5%25EC%259B%2590&kind=Z3&cctvip=undefined&cctvch=undefined&id=74294/CSyslP8ww9ZjsWbRs/mVXsou2biZs46K1abb2%2BLcyFL5dzrTpMsZow/EMg7wNmtCJ1kybp3nMw45d9A%2B/JXgLGI4XkjcbtGjjyOhs5E6lBw=&cctvpasswd=undefined&cctvport=undefined&minX=128.87042157022188&minY=36.07373857956089&maxX=129.00007823849484&maxY=36.138103374399165"

# 저장 경로
UPLOAD_DIR = "./upload"
FILENAME = "cctv_capture.png"
SAVE_PATH = os.path.join(UPLOAD_DIR, FILENAME)

def capture_cctv():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1280x720")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(CCTV_URL)
        time.sleep(3)  # 스트리밍 로드 대기

        driver.save_screenshot(SAVE_PATH)

        # 필요하면 크롭
        img = Image.open(SAVE_PATH)
        cropped = img.crop((550, 100, 1000,  400))  # (left, top, right, bottom)
        cropped.save(SAVE_PATH)

        print("Content-Type: text/html; charset=utf-8\n")
        print("<html><body>")
        print("<h2>CCTV uploading....</h2>")
        print("<meta http-equiv='refresh' content='1; url=check.py?img=cctv_capture.png'>")
        print("</body></html>")

    finally:
        driver.quit()

if __name__ == "__main__":
    capture_cctv()

