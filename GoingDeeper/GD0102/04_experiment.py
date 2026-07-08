import importlib
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm


# 숫자로 시작하는 파일명은 일반 import 문으로 불러올 수 없어 importlib를 사용합니다.
tokenizer_module = importlib.import_module("02_train_tokenizer")
model_module = importlib.import_module("03_model")

get_tokenizer = tokenizer_module.get_tokenizer
SentimentGRU = model_module.SentimentGRU


@dataclass
class ExperimentConfig:
    """
    실험 하이퍼파라미터를 모아둔 설정 객체입니다.
    사용 위치: Movie_review.ipynb, run_experiment()
    """

    TOKENIZER_TYPE: str = "KoNLPy"
    KONLPY_TOKENIZER: str = "Mecab"
    SP_VOCAB_SIZE: int = 10000
    SP_MODEL_TYPE: str = "bpe"
    KONLPY_VOCAB_SIZE: int = 10000
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 128
    N_LAYERS: int = 1
    DROPOUT: float = 0.6
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 128
    MAX_LEN: int = 40
    PATIENCE: int = 3
    WEIGHT_DECAY: float = 1e-5
    SEED: int = 42
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    MODEL_SAVE_PATH: str = "best_model.pth"

    def to_dict(self):
        """
        설정을 dict로 변환합니다.
        사용 위치: Notebook에서 설정 확인 또는 기록
        """
        return self.__dict__.copy()

    def __getitem__(self, key):
        """
        cfg["KEY"] 형태 접근을 지원합니다.
        사용 위치: 기존 딕셔너리 기반 CFG 호환
        """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """
        cfg["KEY"] = value 형태 수정을 지원합니다.
        사용 위치: Notebook에서 일부 설정만 바꾸는 비교 실험
        """
        return setattr(self, key, value)


def seed_everything(seed):
    """
    실험 재현성을 위해 random, numpy, torch seed를 고정합니다.
    사용 위치: Movie_review.ipynb 실험 실행 전
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_tensor_dataset(texts, labels, tokenize_fn, max_len):
    """
    문장과 라벨을 TensorDataset으로 변환합니다.
    사용 위치: create_dataloaders()
    """
    features = tokenize_fn(texts, max_len)
    targets = torch.tensor(labels, dtype=torch.float32)
    return TensorDataset(features, targets)


def create_dataloader(dataset, batch_size, shuffle=False):
    """
    TensorDataset을 DataLoader로 변환합니다.
    사용 위치: create_dataloaders()
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_dataloaders(train_set, val_set, test_set, tokenize_fn, cfg):
    """
    train/validation/test DataLoader를 생성합니다.
    사용 위치: run_experiment()
    """
    train_dataset = make_tensor_dataset(
        train_set["document"].tolist(),
        train_set["label"].values,
        tokenize_fn,
        cfg["MAX_LEN"],
    )
    val_dataset = make_tensor_dataset(
        val_set["document"].tolist(),
        val_set["label"].values,
        tokenize_fn,
        cfg["MAX_LEN"],
    )
    test_dataset = make_tensor_dataset(
        test_set["document"].tolist(),
        test_set["label"].values,
        tokenize_fn,
        cfg["MAX_LEN"],
    )

    train_loader = create_dataloader(train_dataset, cfg["BATCH_SIZE"], shuffle=True)
    val_loader = create_dataloader(val_dataset, cfg["BATCH_SIZE"], shuffle=False)
    test_loader = create_dataloader(test_dataset, cfg["BATCH_SIZE"], shuffle=False)
    return train_loader, val_loader, test_loader


def calculate_binary_accuracy(outputs, labels):
    """
    sigmoid(logit) > 0.5 기준으로 correct/total을 계산합니다.
    사용 위치: train_one_epoch(), validate_one_epoch(), evaluate_model()
    """
    preds = torch.sigmoid(outputs.squeeze()) > 0.5
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    """
    모델을 1 epoch 학습하고 train loss/accuracy를 반환합니다.
    사용 위치: run_training_loop()
    """
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct, total = calculate_binary_accuracy(outputs, labels)
        train_correct += correct
        train_total += total

    return train_loss / len(train_loader), train_correct / train_total


def validate_one_epoch(model, val_loader, loss_fn, device):
    """
    validation loss/accuracy를 계산합니다.
    사용 위치: run_training_loop()
    """
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), labels)
            val_loss += loss.item()

            correct, total = calculate_binary_accuracy(outputs, labels)
            val_correct += correct
            val_total += total

    return val_loss / len(val_loader), val_correct / val_total


def evaluate_model(model, test_loader, loss_fn, device):
    """
    test loss/accuracy를 계산합니다.
    사용 위치: run_experiment()에서 best checkpoint 로드 후
    """
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="[Testing]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), labels)
            test_loss += loss.item()

            correct, total = calculate_binary_accuracy(outputs, labels)
            test_correct += correct
            test_total += total

    return test_loss / len(test_loader), test_correct / test_total


def save_checkpoint(model, path):
    """
    validation loss 개선 시 모델 가중치를 저장합니다.
    사용 위치: run_training_loop()
    """
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    """
    best checkpoint를 모델에 로드합니다.
    사용 위치: run_experiment() 최종 평가 직전
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def create_model(vocab_size, cfg):
    """
    SentimentGRU 모델을 생성하고 device로 이동합니다.
    사용 위치: run_experiment()
    """
    return SentimentGRU(
        vocab_size,
        cfg["EMBEDDING_DIM"],
        cfg["HIDDEN_DIM"],
        cfg["N_LAYERS"],
        cfg["DROPOUT"],
    ).to(cfg["DEVICE"])


def run_training_loop(model, train_loader, val_loader, loss_fn, optimizer, cfg):
    """
    학습 루프, validation, early stopping, checkpoint 저장을 수행합니다.
    사용 위치: run_experiment()
    """
    patience_counter = 0
    best_loss = np.inf

    print("\n🚀 모델 학습을 시작합니다...")
    for epoch in range(cfg["EPOCHS"]):
        epoch_number = epoch + 1
        avg_train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            cfg["DEVICE"],
            epoch_number,
        )
        avg_val_loss, val_accuracy = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            cfg["DEVICE"],
        )

        print(
            f"Epoch {epoch_number:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy*100:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy*100:.2f}%"
        )

        if avg_val_loss < best_loss:
            # 마지막 모델이 아니라 validation loss가 가장 낮은 모델을 평가하기 위해 저장합니다.
            best_loss = avg_val_loss
            save_checkpoint(model, cfg["MODEL_SAVE_PATH"])
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["PATIENCE"]:
            print(f"🛑 Early stopping triggered after {epoch_number} epochs.")
            break


def run_experiment(cfg, train_set, val_set, test_set):
    """
    tokenizer 준비부터 최종 test 평가까지 하나의 실험을 실행합니다.
    사용 위치: Movie_review.ipynb tokenizer별 실험 셀
    """
    tokenize_fn, vocab_size = get_tokenizer(cfg, train_set)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_set,
        val_set,
        test_set,
        tokenize_fn,
        cfg,
    )

    model = create_model(vocab_size, cfg)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["LEARNING_RATE"],
        weight_decay=cfg["WEIGHT_DECAY"],
    )

    run_training_loop(model, train_loader, val_loader, loss_fn, optimizer, cfg)

    print(f"\n🧪 최고 성능 모델('{cfg['MODEL_SAVE_PATH']}')로 최종 평가를 시작합니다...")
    load_checkpoint(model, cfg["MODEL_SAVE_PATH"], cfg["DEVICE"])
    avg_test_loss, test_accuracy = evaluate_model(model, test_loader, loss_fn, cfg["DEVICE"])

    print("\n🎉 모델 평가가 완료되었습니다.")
    print("-" * 40)
    print(f"  - 최종 테스트 손실 (Loss): {avg_test_loss:.4f}")
    print(f"  - 최종 테스트 정확도 (Accuracy): {test_accuracy*100:.2f}%")
    print("-" * 40)

    return {"loss": avg_test_loss, "accuracy": test_accuracy}
