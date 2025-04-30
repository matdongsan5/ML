## Flask APP Entry Point File
## 웹서버 구동 파일
## 세션 설정 및 삭제
from flask import Flask
from flask_migrate import Migrate       ## DB관련
from flask_sqlalchemy import SQLAlchemy ## DB관련
import os

# DB관련 설정
# import DAY04.Project.config as config
import config

#DB 제어 인스턴스
DB = SQLAlchemy()
MIGRATE = Migrate()


## Application Factory Function
## 함수이름 create_app
## 매개변수 : 없음
def create_app():
    ## Flask Server 인스턴스 생성
    app = Flask(__name__)
    
    # db관련 초기화 설정 : config.py 파일 읽어서 웹 서버 설정
    app.config.from_object(config)
    
    # DB 초기화 및 연동
    DB.init_app(app)
    MIGRATE.init_app(app, DB)
    
    
    ##  DB 클래스 정의 모듈 ==> flask의 migrate 기능인식 이ㅜ해 추가
    from .models import models
    
    # 라우팅 등록
    from .views import main_view, user_view, question_view, detail_view, answer_view
    app.register_blueprint(main_view.main_bp)
    app.register_blueprint(user_view.user_bp)
    app.register_blueprint(question_view.question_bp)
    app.register_blueprint(detail_view.detail_bp)
    app.register_blueprint(answer_view.answer_bp)
    
    
    ## Flask Server 인스턴스 반환
    return app