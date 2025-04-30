from flask import Flask
from flask import render_template

### 전역변수
APP = Flask(__name__)

### ===> 라우팅(Routing) 기능 함수
## 기본 URL http://127.0.0.1:5000   
""" 
함수이름: index
함수기능: 127.0.0.1:5000/ 요청 처리 view함수
반환결과: html_string
"""
# @APP.route("/")
def index():
    return render_template('index.html')

""" 
함수이름: greet
함수기능: 127.0.0.1:5000/hello 요청 처리 view함수
변수 없음
반환결과: html_string
"""

def hello():
    return "Hello from add_url_rule"


## 라우팅 URL Rule 추가
APP.add_url_rule('/', endpoint='index_page',
                 view_func=index, methods=['GET'])

APP.add_url_rule('/hello', endpoint='hello_page',
                 view_func=hello, methods=['GET'])

## 조건에 따른 실행 처리
if __name__ == '__main__':
    APP.run()
    