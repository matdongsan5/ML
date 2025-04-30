from flask import Flask
from flask import redirect, render_template #함수

### 전역변수
APP = Flask(__name__)

### ===> 라우팅(Routing) 기능 함수
## 기본 URL http://127.0.0.1:5000   
### ===> 라우팅(Routing) 기능 함수들
@APP.route("/")
def index():
    return render_template('index.html')

## 리다이렉트 URL
@APP.route("/userinfo")
def userinfo():
    return redirect('/')

@APP.route("/hello", endpoint = "hello_page")
def hello():
    return hello

@APP.route("/home")
def home():
    return redirect('/')

## --> 조건에 따른 실행 처리
if __name__ == '__main__':
    APP.run()
    