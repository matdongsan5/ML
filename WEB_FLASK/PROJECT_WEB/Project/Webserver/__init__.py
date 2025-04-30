from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import os
from torchvision import transforms, models
import torch
import torch.nn as nn
from PIL import Image

# Flask 앱 만들기
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Webserver/static/uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# SQLAlchemy 초기화
db = SQLAlchemy(app)

# DB 테이블 정의
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    predicted_label = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.now())

# 처음 한 번만 실행해서 테이블 만들기
# with app.app_context():
#     db.create_all()

# 클래스 이름 설정 (너의 train_dataset.classes 순서)
class_names = ['Etc', 'alouatta_palliata', 'erythrocebus_patas']  # 예시야! 네 데이터에 맞게 수정해줘.

# resnet50 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load('./Model/3_vl0.07.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 메인 페이지 (GET)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# 예측 처리 (POST)
@app.route('/', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # 이미지 열기
    image = Image.open(filepath).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)

    pred_class = class_names[pred.item()]
    confidence = probs[0][pred.item()].item() * 100

    # ✅ 예측 결과를 ORM으로 저장
    prediction = Prediction(
        image_path=filepath,
        predicted_label=pred_class,
        confidence=confidence
    )
    db.session.add(prediction)
    db.session.commit()

    return render_template('index.html', filename=file.filename, prediction=pred_class, confidence=confidence)

@app.route('/predictions')
def show_predictions():
    all_predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template('predictions.html', predictions=all_predictions)

# 업로드한 파일 표시
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

# 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
