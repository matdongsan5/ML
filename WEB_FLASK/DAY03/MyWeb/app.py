## Flask APP Entry Point File
## 세션 설정 및 삭제
from flask import Flask
from flask import session, redirect, url_for
from flask import request, render_template
import os

### 전역변수
APP = Flask(__name__)

# Flask 세션을 위한 시크릿 키
# os.urandom(24): 암호학적으로 안전한 랜덤 바이트를 생성
APP.secret_key = f'super-secret-key-{os.urandom(24)}'
# APP.secret_key = 'super-secret-key-987654321!'

### ===> 라우팅(Routing) 기능 함수
## 기본 URL http://127.0.0.1:5000   

## 처리 URL Rule : http://127.0.0.1:5000   URL 요청 처리 부분
## 처리 view 함수 : index

@APP.route("/")
def index():
    if 'username' in session:
        user_name = session.get('username')
        return render_template('user_page.html',name = user_name)
    else:
        return render_template('login.html')
    
    
## 처리 URL Rule : http://127.0.0.1:5000/login URL
## 처리 view 함수 : login()

@APP.route("/login", methods=['POST'])
def login():
    # login.html <form> 태그 아래 <input> 태그 중 
    # name 속성값이 'username'인 <input> 태그에 입력한 값.
    username = request.form.get('username')
    if username:
        session['username'] = username
        ## 로그인 후 세션에 저장 => 사용자 화면이동
        ## redirect(URL_String)
        ## redirerct(url_for(endpoint))
        return redirect(url_for('index'))

## 처리 URL Rule : http://127.0.0.1:5000/lgout URL
## 처리 view 함수 : logout()

@APP.route("/logout")    
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

#-
# 처리 URL Rule : http://127.0.0.1:5000/check-session URL
# 처리 view 함수 : check_session()

@APP.route('/check-session')
def check_session():
    user = session.get('username')
    if user:
        return f'현재 로그인 사용자: {user}'
    else:
        return '로그인한 사용자가 없습니다.'


# ## ----------------
# ## 처리 URL Rule : http://127.0.0.1:5000/set-cookie URL
# ## 처리 view 함수 : set_cookie()
# ## 반환값: response 객체
# ## ----------------
# @APP.route('/set-cookie')
# def set_cookie():
#     resp = make_response("쿠키가 설정되었습니다.")
#     resp.set_cookie('username', 'c_cookie', max_age=60*60) # 1시간
#     return resp


# ## ---
# ## 처리 URL Rule : http://127.0.0.1:5000/get-cookie URL
# ## 처리 view 함수 : get_cookie()
# ##

# @APP.route('/get-cookie')
# def get_cookie():
#     username = request.cookies.get('username')
#     return f"쿠키값: {username}" if username else '쿠키없음'


# ##---
# ## 처리 URL Rule : http://127.0.0.1:5000/del-cookie URL
# ## 처리 view 함수 : del_cookie()
# ##

# @APP.route("/del-cookie")
# def del_cookie():
#     resp = make_response("쿠키가 삭제되었습니다")
#     resp.set_cookie('username',"", expires=0)
#     return resp
    









## 조건에 따른 실행 처리
if __name__ == '__main__':
    APP.run()
    