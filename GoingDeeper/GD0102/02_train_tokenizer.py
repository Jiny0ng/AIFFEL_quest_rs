import torch
from tqdm.notebook import tqdm


def train_sentencepiece(corpus_path, model_prefix, vocab_size, model_type):
    """
    SentencePiece BPE/Unigram 모델을 학습합니다.
    사용 위치: get_tokenizer("SentencePiece")
    """
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        f"--input={corpus_path} --model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} --model_type={model_type}"
    )
    return f"{model_prefix}.model"


def load_sentencepiece_model(model_path):
    """
    SentencePiece model 파일을 Processor로 로드합니다.
    사용 위치: get_tokenizer("SentencePiece")
    """
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor()
    processor.load(model_path)
    return processor


def sentencepiece_encode(processor, corpus, max_len):
    """
    SentencePiece token id 변환 후 고정 길이로 맞춥니다.
    사용 위치: get_tokenizer()의 tokenize_fn, 04_experiment.make_tensor_dataset()
    """
    sequences = []
    for sentence in corpus:
        ids = processor.encode_as_ids(str(sentence))
        ids = pad_or_truncate(ids, max_len)
        sequences.append(ids)
    return torch.tensor(sequences, dtype=torch.long)


def get_konlpy_tokenizer(tokenizer_name):
    """
    KoNLPy 형태소 분석기(Mecab/Okt)를 생성합니다.
    사용 위치: get_tokenizer("KoNLPy")
    """
    if tokenizer_name == "Mecab":
        from konlpy.tag import Mecab

        return Mecab()
    if tokenizer_name == "Okt":
        from konlpy.tag import Okt

        return Okt()
    raise ValueError("지원하지 않는 토크나이저입니다: Mecab, Okt 중 선택하세요.")


def build_konlpy_vocab(tokenizer, train_corpus, vocab_size):
    """
    KoNLPy 토큰 빈도 기반 vocab을 생성합니다.
    사용 위치: get_tokenizer("KoNLPy")
    재현 규칙: <PAD>=0, <UNK>=1, 나머지는 빈도순으로 선택
    """
    word_counts = {}
    for sentence in tqdm(train_corpus, desc="KoNLPy Freq. Counting"):
        tokens = tokenizer.morphs(str(sentence))
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1

    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    top_words = sorted_words[: vocab_size - 2]

    word_index = {"<PAD>": 0, "<UNK>": 1}
    for word in top_words:
        word_index[word] = len(word_index)
    return word_index


def konlpy_encode(tokenizer, word_index, corpus, max_len):
    """
    KoNLPy 형태소를 token id로 바꾸고 고정 길이로 맞춥니다.
    사용 위치: get_tokenizer()의 tokenize_fn, 04_experiment.make_tensor_dataset()
    """
    sequences = []
    for sentence in corpus:
        tokens = tokenizer.morphs(str(sentence))
        ids = [word_index.get(token, 1) for token in tokens]
        ids = pad_or_truncate(ids, max_len)
        sequences.append(ids)
    return torch.tensor(sequences, dtype=torch.long)


def pad_or_truncate(ids, max_len, pad_id=0):
    """
    token id 시퀀스를 max_len으로 자르거나 pad_id로 채웁니다.
    사용 위치: sentencepiece_encode(), konlpy_encode()
    """
    return ids[:max_len] if len(ids) > max_len else ids + [pad_id] * (max_len - len(ids))


def get_tokenizer(cfg, train_df):
    """
    cfg에 맞는 tokenize_fn과 vocab_size를 반환합니다.
    사용 위치: 04_experiment.run_experiment()
    """
    tokenizer_type = cfg["TOKENIZER_TYPE"]

    if tokenizer_type == "SentencePiece":
        corpus_path = "nsmc_corpus.txt"
        model_prefix = f'nsmc_{cfg["SP_MODEL_TYPE"]}_{cfg["SP_VOCAB_SIZE"]}'
        # SentencePiece는 파일 입력을 받으므로 train 문장을 corpus로 저장합니다.
        train_df["document"].to_csv(corpus_path, index=False, header=False)

        model_path = train_sentencepiece(
            corpus_path=corpus_path,
            model_prefix=model_prefix,
            vocab_size=cfg["SP_VOCAB_SIZE"],
            model_type=cfg["SP_MODEL_TYPE"],
        )
        processor = load_sentencepiece_model(model_path)
        vocab_size = processor.get_piece_size()

        def tokenize_fn(corpus, max_len):
            return sentencepiece_encode(processor, corpus, max_len)

        print(f"✅ SentencePiece 토크나이저 준비 완료 (vocab_size: {vocab_size})")
        return tokenize_fn, vocab_size

    if tokenizer_type == "KoNLPy":
        tokenizer = get_konlpy_tokenizer(cfg["KONLPY_TOKENIZER"])
        word_index = build_konlpy_vocab(
            tokenizer=tokenizer,
            train_corpus=train_df["document"],
            vocab_size=cfg["KONLPY_VOCAB_SIZE"],
        )
        vocab_size = len(word_index)

        def tokenize_fn(corpus, max_len):
            return konlpy_encode(tokenizer, word_index, corpus, max_len)

        print(
            f"✅ KoNLPy({cfg['KONLPY_TOKENIZER']}) 토크나이저 준비 완료 "
            f"(vocab_size: {vocab_size})"
        )
        return tokenize_fn, vocab_size

    raise ValueError("TOKENIZER_TYPE은 'SentencePiece' 또는 'KoNLPy'여야 합니다.")
