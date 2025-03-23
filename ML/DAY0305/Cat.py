#-----------------------------------------------------------------------------
# 고양이 데이터를 표현/저장하는 클래스 정의/설계
#-----------------------------------------------------------------------------
# * 특징/속성/성질/외형 : 품종, 털색상, 눈색상, 무게, 나이, 성별, 이름
# * 행동/기능/동작      : 울기, 긁기, 올라가기
#----------------------------------------------------------------------------
# 클래스이름 : Cat
# 클래스속성 : kind, coat_color, eye_color, weight, age, gender, name
# 클래스기능 : break(), bite(), tailing()

class Cat:
    ## 객체 즉, 인스턴스 생성 시 Dog 마다 생성되는 속성
    ## 파이썬 시스템에 의해 실행되는 메서드 : 콜백(callback) 메서드/함수
    
    def __init__(self, kind, coat_color, eye_color, weight, age, gender, name):
        self.kind = kind
        self.coat_color =coat_color
        self.eye_color = eye_color
        self.weight = weight
        self.age = age
        self.gender = gender
        self.name = name
        print("Cat.__init__",self)

    
    ## 메서드 이름 : bark()
    ## 매개변수 : 없음
    ## 메서드결과 : 없음
    def bark(self):
        print(f"{self.name}가 짖는다")
    
    def bite(self,what):
        self.what=what
        print(f"{self.name}가 {self.what}을 문다")
            
    
#------------------------------------------------------
# 객체 생성
#   생성 메서드 이름 : 클래스이름() ---> 클래스 내부으의 __init__() 호출
#-------------------------------------------------------
myDog = Dog('코숏', '검은색', '검은색', 20, 2, '남자', '코숏' )
myDog1 = Dog('앙골라', '흰색', '검은색', 30, 3, '여자', '앙골라' )
myDog2 = Dog('잡종', '회색', '검은색', 10, 4, '남자', '잡종' )
print(id(myDog))

myDog.bark()
myDog.bite('도둑')
##
## 객체의 속성과 메서드 사용하기
## 속성 읽기    객체변수명.속성명
## 속성 변경    객체변수명.속성명 = 새값
## 메서드호출   객체변수명.메서드명()

myDog.age = 3
print(myDog.age)
    
a = set(1,2,3)
b = set(1,2,3,4,5)
print(a.intersection(b))