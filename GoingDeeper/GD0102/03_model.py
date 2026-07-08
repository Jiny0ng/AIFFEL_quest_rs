import torch.nn as nn


class SentimentGRU(nn.Module):
    """
    NSMC 영화 리뷰 감정분석용 GRU 모델입니다.
    사용 위치: 04_experiment.create_model()
    구조: Embedding -> GRU -> Dropout -> Linear(logit)
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        """
        모델 layer를 초기화합니다.
        사용 위치: SentimentGRU 객체 생성 시 자동 호출
        """
        super(SentimentGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        # 기존 통합 실험 블록과 동일하게 GRU 마지막 출력 뒤에만 dropout을 적용합니다.
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        token id tensor를 이진 분류 logit으로 변환합니다.
        사용 위치: train_one_epoch(), validate_one_epoch(), evaluate_model()
        """
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        # 기존 실험과 같이 마지막 time step 출력을 문장 표현으로 사용합니다.
        last_output = gru_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output
