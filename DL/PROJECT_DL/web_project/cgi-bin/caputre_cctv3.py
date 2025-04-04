
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
CCTV_URL = "https://www.utic.go.kr/jsp/map/cctvStream.jsp?cctvid=E901850&cctvname=%255B%25EA%25B5%25AD%25EB%258F%258428%255D%25EC%2598%2581%25EC%25B2%259C%25EC%25B2%25AD%25EC%25A0%2595%25EA%25B5%2590%25EC%25B0%25A8%25EB%25A1%259C%25EB%258F%2599%25EC%25B8%25A1&kind=Z3&cctvip=undefined&cctvch=undefined&id=72739/FO2vzwVBD9axlZVsJXZHMwoWGMyKhrorZbDhrrt8tWhJuuCb7AKdmt2li38%2BfnOMtAONAMBPbdBoM2ixTlKSxPiDPkkVROxNHTRnowEaVd0=&cctvpasswd=undefined&cctvport=undefined&minX=129.0825398849302&minY=35.948409046033525&maxX=129.21213771840365&maxY=36.01256455212556"

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

