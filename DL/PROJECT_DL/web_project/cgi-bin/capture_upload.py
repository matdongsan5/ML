#!/usr/bin/env python3
import os
import subprocess
import cgi
import cgitb

cgitb.enable()

# CCTV 촬영 스크립트 실행
subprocess.run(["python3", "capture_cctv.py"])

# check.py로 업로드한 파일 전달
print("Content-Type: text/html; charset=utf-8\n")
print("<html><body>")
print("<h2>CCTV screenshot uploading...</h2>")
print("<meta http-equiv='refresh' content='1; url=check.py'>")  # 1초 후 리다이렉트
print("</body></html>")
