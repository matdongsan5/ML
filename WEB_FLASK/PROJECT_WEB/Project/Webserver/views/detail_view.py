## --------------------------------------------------------
## APP MODULIZATION - Main Module
## --------------------------------------------------------
## 모듈 로딩
## --------------------------------------------------------
from flask import Blueprint , render_template
from ..models.models import Question

## --------------------------------------------------------
## 모듈 인스턴스 생성 
## --------------------------------------------------------
## 매개변수 - BP name       : 블루프린트 인스턴스 이름
##           import_name   : 블루프린트가 정의된 현재 모듈 이름
##           url_prefix    : 기본 URL을 생략한 시작 URL
## --------------------------------------------------------
detail_bp=Blueprint('DETAIL', 
                  __name__, 
                  template_folder='templates',
                  url_prefix='/detail')

## --------------------------------------------------------
## URL rule 등록 By Decorator
## - 기본 URL : http://127.0.0.1:5000/detail/id
## --------------------------------------------------------
@detail_bp.route('/<int:question_id>/')
def detail(question_id):
    question = Question.query.get(question_id)
    return render_template('question_detail.html', question=question)