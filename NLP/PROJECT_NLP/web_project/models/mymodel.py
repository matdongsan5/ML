import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## 클래스이름 : TextModelComplex
## 부모클래스 : Module
## 매개변수   : 단어사전 갯수(vocab_size), 임베딩 차원(embed_dim),
##              중간 은닉층 차원(hidden_dim), 분류 클래스 갯수(num_class),
##              드롭아웃 확률(dropout_p)
## -------------------------------------------------------------------------
class TextModelComplex(nn.Module):
    ## 모델 층 정의 메서드 --------------------------------------
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, dropout_p=0.3):
        """
        모델의 레이어를 초기화합니다.

        Args:
            vocab_size (int): 단어 사전의 크기.
            embed_dim (int): 각 단어 벡터의 임베딩 차원.
            hidden_dim (int): 중간 은닉층의 차원.
            num_class (int): 출력 클래스의 개수.
            dropout_p (float, optional): 드롭아웃 확률. 기본값은 0.5.
        """
        super().__init__() # 부모 클래스(nn.Module)의 __init__ 호출

        ## 1. 임베딩 레이어: 단어 ID를 벡터로 변환 (EmbeddingBag 사용)
        # 고차원 희소 벡터(단어 ID) ==> 저차원 밀집 벡터 (단어 임베딩)
        # EmbeddingBag은 각 시퀀스 내 임베딩 벡터들의 평균/합 등을 계산합니다.
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)

        ## 2. 첫 번째 Linear 레이어 (임베딩 -> 은닉층)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)

        ## 3. 활성화 함수 (비선형성 추가)
        self.relu = nn.ReLU()

        ## 4. Dropout 레이어 (과적합 방지)
        self.dropout = nn.Dropout(dropout_p)

        ## 5. 두 번째 Linear 레이어 (은닉층 -> 출력 클래스)
        # 다중 분류를 위한 최종 출력 레이어
        self.fc2 = nn.Linear(hidden_dim, num_class)

        ## 초기 가중치 설정 => self.메서드이름() : 같은 클래스에 존재하는 메서드 호출
        self.init_weights()

    ## 가중치 초기화 기능의 메서드 ---------------------------
    def init_weights(self):
        """
        모델의 가중치를 균등 분포로 초기화합니다.
        """
        initrange = 0.5
        # 임베딩 레이어 가중치 초기화
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # 첫 번째 Linear 레이어 가중치 및 편향 초기화
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        # 두 번째 Linear 레이어 가중치 및 편향 초기화
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    ## 순전파 학습 메서드 -------------------------------------------
    def forward(self, text, offsets):
        """
        입력 데이터를 받아 모델의 순전파를 수행합니다.

        Args:
            text (Tensor): 텍스트 데이터 텐서 (보통 1차원).
            offsets (Tensor): 각 시퀀스의 시작 인덱스를 나타내는 텐서.

        Returns:
            Tensor: 모델의 최종 출력 (클래스별 로짓).
        """
        ## 1. 임베딩 적용 (EmbeddingBag)
        # 입력 텍스트와 오프셋을 사용하여 각 시퀀스의 임베딩 벡터 평균/합 계산
        # 결과: [batch_size, embed_dim]
        embedded = self.embedding(text, offsets)

        ## 2. 첫 번째 Linear 레이어 통과
        # 결과: [batch_size, hidden_dim]
        hidden = self.fc1(embedded)

        ## 3. ReLU 활성화 함수 적용
        hidden = self.relu(hidden)

        ## 4. Dropout 적용 (훈련 시에만 동작)
        hidden = self.dropout(hidden)

        ## 5. 두 번째 Linear 레이어 통과 (최종 출력)
        # 결과: [batch_size, num_class]
        # 다중 분류이므로 손실 함수 (예: CrossEntropyLoss)에서 내부적으로 softmax 처리를 합니다.
        output = self.fc2(hidden)
        return output