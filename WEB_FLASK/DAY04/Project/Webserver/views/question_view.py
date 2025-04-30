## --------------------------------------------------------
## APP MODULIZATION - Main Module
## --------------------------------------------------------
## 모듈 로딩
## --------------------------------------------------------
from flask import Blueprint 
from flask import render_template, request,redirect, url_for
from ..models.models import Question
from Webserver import DB 
from datetime import datetime

## --------------------------------------------------------
## 모듈 인스턴스 생성 
## --------------------------------------------------------
## 매개변수 - BP name       : 블루프린트 인스턴스 이름
##           import_name   : 블루프린트가 정의된 현재 모듈 이름
##           url_prefix    : 기본 URL을 생략한 시작 URL
## --------------------------------------------------------
question_bp=Blueprint('QUESTION', 
                  __name__, 
                  template_folder='templates',
                  url_prefix='/question')

## --------------------------------------------------------
## URL rule 등록 By Decorator
## - 기본 URL : http://127.0.0.1:5000/question/list
## --------------------------------------------------------
@question_bp.route('/list')
def question_list():
    ## 등록 날짜별 내림차순 정렬한 질문 객체 리스트
    question_list = Question.query.order_by(Question.create_data.desc())
    ## WEB에 출력하기 위해 전달.
    return render_template('question_list.html', question_list=question_list)

## --------------------------------------------------------
## - 기본 URL : http://127.0.0.1:5000/question/list
## --------------------------------------------------------
@question_bp.route('/create', methods=('POST',))
def question_create():
    # 질문 로딩
    subject = request.form['subject']
    content = request.form['content']
    # 답변 생성
    answer = Question(subject=subject, content=content, create_data=datetime.now())
    # DB 적용
    DB.session.add(answer)
    DB.session.commit()
    return redirect(url_for('QUESTION.question_list'))