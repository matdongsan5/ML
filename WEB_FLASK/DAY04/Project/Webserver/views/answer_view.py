## --------------------------------------------------------
## APP MODULIZATION - Main Module
## --------------------------------------------------------
## 모듈 로딩
## --------------------------------------------------------
from flask import Blueprint 
from flask import render_template,request, redirect, url_for
from ..models.models import Question, Answer
from datetime import datetime
from Webserver import DB

## --------------------------------------------------------
## 모듈 인스턴스 생성 
## --------------------------------------------------------
## 매개변수 - BP name       : 블루프린트 인스턴스 이름
##           import_name   : 블루프린트가 정의된 현재 모듈 이름
##           url_prefix    : 기본 URL을 생략한 시작 URL
## --------------------------------------------------------
answer_bp=Blueprint('ANSWER', 
                  __name__, 
                  template_folder='templates',
                  url_prefix='/answer')

## --------------------------------------------------------
## URL rule 등록 By Decorator
## - 기본 URL : http://127.0.0.1:5000/answer/list
## --------------------------------------------------------
@answer_bp.route('/create/<int:question_id>', methods=('POST',))
def create(question_id):
    # 질문 로딩
    question = Question.query.get_or_404(question_id)
    content = request.form['content']
    # 답변 생성
    answer = Answer(question=question, content=content, create_data=datetime.now())
    # DB 적용
    DB.session.add(answer)
    DB.session.commit()
    return redirect(url_for('DETAIL.detail', question_id=question_id))