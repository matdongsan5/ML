from flask import Flask, render_template
from random import sample

app = Flask(__name__)

@app.route('/')
def main():
    return 'hello world'

# 사람 수 만큼 점심 메뉴 추천
@app.route("/lunch/<int:people>")
def lunch(people):
  menu = ["짜장면", "짬뽕", "라면", "브리또", "사과", "찜닭"]
  return f'{sample(menu, people)}'


@app.route("/show")
def show():
  # 음식 사진을 static 폴더에 추가하고 menu에 집어 넣습니다
  # 음식 메뉴 개수는 더 많아도 됩니다
  menu = ['000101.png', '000201.png']
  # 음식 메뉴 1개를 뽑습니다
  # pickme = ''.join(sample(menu,1))
  pickme = menu[0]
  # index.html 파일에 이미지를 불러옵니다
  return render_template('index.html', food_img='seals/'+pickme)
  # return render_template('index.html', food_img='seals/000101.png')


# if __name__ == "__main__": 
#     app.run()           # 객체의 run함수를 이용하여 로컬 서버에서 앱 실행
if __name__ == "__main__":
    app.run(debug=True)     #debugMode 실행 명령어