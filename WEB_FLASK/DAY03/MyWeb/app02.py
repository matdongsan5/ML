## Flask APP Entry Point File
## Cookie 설정 및 사용
from flask import Flask
from flask import render_template, make_response, request

### 전역변수
APP = Flask(__name__)

### ===> 라우팅(Routing) 기능 함수
## 기본 URL http://127.0.0.1:5000   

## 라우팅 URL Rule 등록 by decorator
""" 
함수이름: index
함수기능: 127.0.0.1:5000/ 요청 처리 view함수
반환결과: html_string
"""
@APP.route("/")
def index():
    return render_template('index.html')


## ----------------
## 처리 URL Rule : http://127.0.0.1:5000/set-cookie URL
## 처리 view 함수 : set_cookie()
## 반환값: response 객체
## ----------------
@APP.route('/set-cookie')
def set_cookie():
    resp = make_response("쿠키가 설정되었습니다.")
    resp.set_cookie('username', 'c_cookie', max_age=60*60) # 1시간
    return resp


## ---
## 처리 URL Rule : http://127.0.0.1:5000/get-cookie URL
## 처리 view 함수 : get_cookie()
##

@APP.route('/get-cookie')
def get_cookie():
    username = request.cookies.get('username')
    return f"쿠키값: {username}" if username else '쿠키없음'


##---
## 처리 URL Rule : http://127.0.0.1:5000/del-cookie URL
## 처리 view 함수 : del_cookie()
##

@APP.route("/del-cookie")
def del_cookie():
    resp = make_response("쿠키가 삭제되었습니다")
    resp.set_cookie('username',"", expires=0)
    return resp
    









## 조건에 따른 실행 처리
if __name__ == '__main__':
    APP.run()
    