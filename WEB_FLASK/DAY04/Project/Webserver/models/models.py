# --------------------------------------------------------------------------------
# Flask Framework에서 Database의 Table 맵핑 class 정의 
# - 파일명 : models.py
# - 예  시 : database내에 question 테이블 ==> Question 클래스
#                        answer  테이블   ==> Answer 클래스
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# 데이터베이스의 테이블 정의 클래스
# --------------------------------------------------------------------------------
# 모듈 로딩
from Webserver import DB

# --------------------------------------------------------------------------------
# Question 테이블 정의 클래스
# - PK : id
# --------------------------------------------------------------------------------
class Question(DB.Model):
    # 컬럼 정의
    id = DB.Column(DB.Integer, primary_key=True)
    subject = DB.Column(DB.String(200), nullable=False)
    content = DB.Column(DB.Text(), nullable=False)
    create_data = DB.Column(DB.DateTime(), nullable=False)

# --------------------------------------------------------------------------------
# Answer 테이블 정의 클래스
# - PK : id
# - FK : question.id
# --------------------------------------------------------------------------------
class Answer(DB.Model):
    # 컬럼 정의
    id = DB.Column(DB.Integer, primary_key=True)
    question_id = DB.Column(DB.Integer,
                            DB.ForeignKey('question.id', ondelete='CASCADE'))
    question = DB.relationship('Question', backref=DB.backref('answer_set', ))
    content = DB.Column(DB.Text(), nullable=False)
    create_data = DB.Column(DB.DateTime(), nullable=False)