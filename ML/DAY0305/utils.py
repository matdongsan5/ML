# ## 평면에 점을 그려주는 프로그램 설계

# - 사용자가 원하는 위치에 점을 표시

# - 점을 표현/저장하는 데이터 타입 필요

# - 새로운 데이터 타입 설계
#     * 속성/특징/외형: x, y, color, 반지름 ==> 코드화 ==> 변수명 x y color r
#     * 행동/기능/역할 : 점그리기, 정보출력 -->           메서드명 drawing(), printinfo()
#     * 데이터타입이름 : Point
#     * 부모/슈퍼클래스

#     * 속성/특징/외형: x, y, color, 반지름
#     * 행동/기능/역할 : 점그리기, 정보출력
#     * 데이터타입이름 : Point
#     * 부모/슈퍼클래스


class Point:
    
    ## 객체/인스턴스 생성 및 속성 초기화 매서드
    # self :  힙 메모리에 저장되는 주소 저장 매개변수
    def __init__(self, x, y, color, r):
        self.x = x
        self.y = y
        self.color = color
        self.r = r

    #클래스 전용 함수 즉, 메서드
    #메서드 이름 : drawing
    #매개변수들 : 
    #매서드결과: 반환값없음
    
    def drawing(self):
        #파란점
        print(f"{self.color} *")
        
    #메서드 이름 : printinfo
    #매개변수들 : 
    #매서드결과: 반환값없음
    def printinfo(self):
        
        print(f"위치: {self.x}, {self.y}")
        print(f"크기: {self.r}")
        print(f"색상: {self.color}")
        
        
#  * 속성/특징/외형: x, y, color, 반지름 ==> 코드화 ==> 변수명 x y color r
#     * 행동/기능/역할 : 점그리기, 정보출력 -->           메서드명 drawing(), printinfo()
#     * 데이터타입이름 : Point3D
#     * 부모/슈퍼클래스


class Point3D(Point):
    ## 객체/인스턴스 생성 및 속성 초기화 매서드
    # self :  힙 메모리에 저장되는 주소 저장 매개변수
    def __init__(self, x, y, z, color, r):
        super().__init__(x, y, color, r)
        self.z = z
        
blackPoint = Point(10,10,'black',5)
redPoint = Point3D(5,5,5, 'red', 5)

blackPoint.drawing()
redPoint.drawing()

redPoint.r = 10
redPoint.printinfo()

blackPoint.printinfo() 

print(__name__)