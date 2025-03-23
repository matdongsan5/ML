#-----------------------------------------------------------------------------
# 고양이 데이터를 표현/저장하는 클래스 정의/설계
#-----------------------------------------------------------------------------
# * 특징/속성/성질/외형 : 품종, 털색상, 눈색상, 무게, 나이, 성별, 이름
# * 행동/기능/동작      : 울기, 긁기, 올라가기
#----------------------------------------------------------------------------
# 클래스이름 : Cat
# 부모클래스 : Animal
# 클래스속성 : kind, coat_color, eye_color, weight, age, gender, name
# 클래스기능 : break(), bite(), tailing()
from Animal import *

class Cat2(Animal):
    ## 객체 즉, 인스턴스 생성 시 Dog 마다 생성되는 속성
    ## 파이썬 시스템에 의해 실행되는 메서드 : 콜백(callback) 메서드/함수
    
    def __init__(self, kind, coat_color, eye_color, weight, age, gender, name):
        super().__init__(kind, coat_color, eye_color, weight, age, gender, name)
        print("Cat.__init__",self)
   
    ## 메서드 이름 : bark()
    ## 매개변수 : 없음
    ## 메서드결과 : 없음
    def bark(self):
        print(f"{self.name}가 짖는다")
    
    def bite(self,what):
        self.what=what
        print(f"{self.name}가 {self.what}을 문다")
    
    ## 메서드 이름 : tailing()
    ## 매개변수 : 없음
    ## 메서드결과 : 없음
    def tailing(self):
        print(f"{self.name}가 꼬리로 바닥을 친다.")