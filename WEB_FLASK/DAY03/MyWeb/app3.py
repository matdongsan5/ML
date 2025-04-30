## Flask APP Entry Point File
## 세션 설정 및 삭제
from flask import Flask
from flask import render_template
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
    
        return render_template('login.html')
    
 
 
 
 
    









## 조건에 따른 실행 처리
if __name__ == '__main__':
    APP.run()
    