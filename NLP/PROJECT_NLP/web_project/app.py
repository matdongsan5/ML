""" 
필요한 것
저장한 모델
models안의 mymodel.py
저장한 vocab

"""


from flask import Flask, request, render_template
import torch
import pickle
from models.mymodel import TextModelComplex

app = Flask(__name__)

import pickle

with open("./models/key_to_index.pkl", "rb") as f:
    key_to_index = pickle.load(f)

## 학습 설정
INPUT_SIZE      = 256
LR              = 0.01
EPOCHS          = 10
STEP_SIZE       = 5
NUM_CLASS       = 4

EMBEDDING_DIM   = 128
HIDDEN_DIM      = 128
# VOCAB_SIZE      = len(VOCAB)
VOCAB_SIZE      = len(key_to_index)

# 모델, 토크나이저 불러오기
# 1. 모델 구조 먼저 정의
## 필요한 변수 가져와서 같이 정의해야함
model = TextModelComplex(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASS)

# 2. state_dict 불러오기
state_dict = torch.load("./models/sd_9_v57.80.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# 3. 평가 모드 전환
model.eval()

from konlpy.tag import Okt
tokenizer = Okt()

# 텍스트 전처리 및 텐서 변환 함수
MAX_LEN = 50  # 예시
PAD_IDX = 0
import string
PUNC = string.punctuation
def preprocess(text):
    text = ''.join([x for x in text if x not in PUNC ])
    tokens = tokenizer.morphs(text)
    indices = [key_to_index.get(token, key_to_index.get("<UNK>", 1)) for token in tokens]
    if len(indices) < MAX_LEN:
        indices += [PAD_IDX] * (MAX_LEN - len(indices))
    else:
        indices = indices[:MAX_LEN]
    input_tensor = torch.tensor([indices])  # [1, MAX_LEN]
    return input_tensor

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None  # 예측 결과를 저장할 변수

    if request.method == "POST":
        input_text = request.form.get("text", "")
        input_tensor = preprocess(input_text)

        with torch.no_grad():
            output = model(input_tensor, None)
            pred = torch.argmax(output, dim=1).item()
        
        label_map = {0: 80, 1: 90, 2: '00', 3: 10}
        prediction = label_map.get(pred, "알 수 없는 예측 결과")  # 예측 결과를 label_map에 따라 변환

    # render_template를 사용하여 HTML로 예측 결과 전달
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)