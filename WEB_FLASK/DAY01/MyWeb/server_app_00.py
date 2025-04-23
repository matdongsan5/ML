from flask import Flask
from flask import redirect #함수

### 전역변수
APP = Flask(__name__)

### ===> 라우팅(Routing) 기능 함수
## 기본 URL http://127.0.0.1:5000   
### ===> 라우팅(Routing) 기능 함수들
@APP.route("/")
def index():
    return "INDEX WEB PAGE"

## 변수 URL
@APP.route("/<msg>")
def message(msg):
    return f"Request URL : /{msg}"

## 리다이렉트 URL
@APP.route("/userinfo")
def userinfo():
    # return APP.redirect('/')
    return redirect('/')


@APP.route("/<test>")
def test2(test):
    return f"TEST :{test} ~~"

@APP.route("/<int:number>")
def test3(number):
    return f"Select Number :{number} !"

@APP.route("/3")
def hello():
    return """Request URL : /3
            <html>
                <head>
                </head>
                <body>
                    <div background-color=orange>html test </div>
                </body>
            </html
"""

## --> 조건에 따른 실행 처리
if __name__ == '__main__':
    APP.run()
    