## --------------------------------------------------------
## APP MODULIZATION - Main Module
## --------------------------------------------------------
## 모듈 로딩
## --------------------------------------------------------
from flask import Blueprint, render_template

## --------------------------------------------------------
## 모듈 인스턴스 생성 
## --------------------------------------------------------
## 매개변수 - BP name       : 블루프린트 인스턴스 이름
##           import_name   : 블루프린트가 정의된 현재 모듈 이름
##           url_prefix    : 기본 URL을 생략한 시작 URL
## --------------------------------------------------------
prediction_bp=Blueprint('prediction', 
                  __name__, 
                  template_folder='templates',
                  url_prefix='/prediction')

## --------------------------------------------------------
## URL rule 등록 By Decorator
## - 기본 URL : http://127.0.0.1:5000/prediction
## --------------------------------------------------------
@prediction_bp.route('/')
def index():
    return render_template('predictions.html')