#모듈로딩
from Animal import *
from Cat2 import *
from Dog2 import *


myDog = Dog2('차우차우', '검은색', '검은색', 20, 2, '남자', '차우' )
yourDog = Dog2('차차', '검은색', '검은색', 20, 2, '남자', '우123' )
myCat = Cat2('코숏', '검은색', '검은색', 20, 2, '남자', '코숏' )
yourCat = Cat2('앙골라', '흰색', '검은색', 30, 3, '여자', '앙골라' )


print(f"myDog.name {myDog.name}") 
print(f"myDog.weight {myDog.weight}") 
print(f"myDog.age {myDog.age}") 


myCat.tailing()
myDog.tailing()
