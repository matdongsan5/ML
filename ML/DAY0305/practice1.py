#-----------------------------------------------------------------------------
# 사람 데이터를 표현/저장하는 클래스 정의/설계
#-----------------------------------------------------------------------------
# * 특징/속성/성질/외형 : 이름, 키, 나이
# * 행동/기능/동작      : 먹기, 자기, 걷기
#----------------------------------------------------------------------------
# 클래스이름 : Human
# 클래스속성 : name, height, age
# 클래스기능 : eat(), sleep(), walk()

class Human:
    def __init__(self, name, height, age):
        self.name = name
        self.height = height
        self.age= age
    
    def eat(self):
        print(f"{self.name}가 먹는다")
    
    def sleep(self):
        print(f"{self.name}이/가 잔다")
    
    def walk(self):
        print(f"{self.name}이/가 걷는다")    

#-----------------------------------------------------------------------------
# 학생 데이터를 표현/저장하는 클래스 정의/설계
#-----------------------------------------------------------------------------
# * 특징/속성/성질/외형 : 이름, 키, 나이
# * 행동/기능/동작      : 먹기, 자기, 걷기, 공부하기
#----------------------------------------------------------------------------
# 부모클래스 : Human
# 클래스이름 : Student
# 클래스속성 : name, height, age
#               grade                
# 클래스기능 : eat(), sleep(), walk()        
#               study()


class Student(Human):
    def __init__(self, name, height, age, grade):
        super().__init__(name, height, age)
        self.grade = grade
        
    def study(self):
        print(f"{self.name}이/가 공부한다")
        
    def sleep(self):
        print(f"{self.name}이/가 수업중에 잔다")
        
    
#-----------------------------------------------------------------------------
# 의사 데이터를 표현/저장하는 클래스 정의/설계
#-----------------------------------------------------------------------------
# * 특징/속성/성질/외형 : 이름, 키, 나이
# * 행동/기능/동작      : 먹기, 자기, 걷기, 수술하기
#----------------------------------------------------------------------------
# 부모클래스 : Human
# 클래스이름 : Doctor
# 클래스속성 : name, height, age
#               type                
# 클래스기능 : eat(), sleep(), walk()        
#               surgeory()


class Doctor(Human):
    def __init__(self, name, height, age, type):
        super().__init__(name, height, age)
        self.type = type
        
    def surgeory(self):
        print(f"{self.name}이/가 수술한다")
        
    def walk(self):
        print(f"{self.name}은/는 걸어서 출근한다")
        
        
John = Human('John', 180, 30)
Cena = Student('Cena', 175, 20, 4)
Donald = Doctor('Donald', 160, 45, 'brain')

print(f"John.name {John.name}")
print(f"Cena.name {Cena.name}")
print(f"Donald.name {Donald.name}")

John.sleep()
#오버라이딩 sleep()
Cena.sleep()
Cena.study()
John.walk()
#오버라이딩 walk()
Donald.walk() 
Donald.surgeory()
