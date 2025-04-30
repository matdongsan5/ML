## Flask APP Entry Point File
## 세션 설정 및 삭제
from flask import Flask
from flask import session, redirect, url_for
from flask import request, render_template
import os


## Application Factory Function
## 함수이름 create_app
## 매개변수 : 없음
def create_app():
    ## Flask Server 인스턴스 생성
    app = Flask(__name__)
    
    # from.views import data_view
    # app.register_blueprint(data.view.data_BP)
    
    # 라우팅 등록
    from .views import main_view, user_view
    app.register_blueprint(main_view.main_bp)
    app.register_blueprint(user_view.user_bp)
    
    
    ## Flask Server 인스턴스 반환
    return app