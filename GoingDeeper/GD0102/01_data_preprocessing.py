import os

import pandas as pd


# 기존 노트북의 NSMC 기본 경로입니다. DATA_DIR을 넘기면 이 값은 사용하지 않습니다.
DEFAULT_DATA_DIR = os.path.join(
    os.getenv("HOME"),
    "work",
    "workplace",
    "AIFFEL_quest_rs",
    "Exploration",
    "Quest05",
    "sentiment_classification",
    "data",
)


def load_nsmc_data(data_dir=None):
    """
    NSMC train/test TSV 파일을 불러옵니다.
    사용 위치: load_and_preprocess_data()
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    train_path = os.path.join(data_dir, "ratings_train.txt")
    test_path = os.path.join(data_dir, "ratings_test.txt")

    train_data = pd.read_table(train_path)
    test_data = pd.read_table(test_path)
    return train_data, test_data


def remove_missing_documents(df):
    """
    document 컬럼이 결측치인 행을 제거합니다.
    사용 위치: clean_nsmc_dataframe()
    """
    return df.dropna(subset=["document"]).copy()


def remove_duplicate_documents(df):
    """
    document 컬럼 기준으로 중복 리뷰를 제거합니다.
    사용 위치: clean_nsmc_dataframe()
    """
    return df.drop_duplicates(subset=["document"]).copy()


def clean_nsmc_dataframe(df):
    """
    결측치 제거 후 document 중복을 제거합니다.
    사용 위치: load_and_preprocess_data()
    """
    cleaned_df = remove_missing_documents(df)
    cleaned_df = remove_duplicate_documents(cleaned_df)
    return cleaned_df


def split_train_validation(train_df, test_size=0.2, seed=42):
    """
    train 데이터를 train/validation으로 나눕니다.
    사용 위치: load_and_preprocess_data()
    재현 규칙: seed 고정, label 비율 유지를 위한 stratify 적용
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        train_df,
        test_size=test_size,
        random_state=seed,
        stratify=train_df["label"],
    )


def save_sentencepiece_corpus(train_df, corpus_path="nsmc_corpus.txt"):
    """
    document 컬럼을 SentencePiece 학습용 corpus로 저장합니다.
    사용 위치: Notebook 또는 02_train_tokenizer.get_tokenizer() 대체 호출
    """
    train_df["document"].to_csv(corpus_path, index=False, header=False)
    return corpus_path


def load_and_preprocess_data(data_dir=None, seed=42, validation_size=0.2):
    """
    NSMC 로드, 정제, train/validation split을 실행합니다.
    사용 위치: Movie_review.ipynb 데이터 준비 셀
    """
    train_data, test_data = load_nsmc_data(data_dir)

    train_data = clean_nsmc_dataframe(train_data)
    test_data = clean_nsmc_dataframe(test_data)

    train_set, val_set = split_train_validation(
        train_data,
        test_size=validation_size,
        seed=seed,
    )

    print("✅ 데이터 로드 및 전처리 완료")
    return train_set, val_set, test_data
