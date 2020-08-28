# coding: UTF-8

import os
import re
import tqdm
import json
import time
import codecs
import torch
import numpy as np
import hao
from torch.utils.data import DataLoader, TensorDataset
from datetime import timedelta


UNK, PAD, CLS = '[UNK]', '[PAD]', '[CLS]'
LOGGER = hao.logs.get_logger(__name__)


class InputFeatures(object):
    def __init__(self, input_id, bigram_id, field_id, input_mask, seq_len, label):
        self.input_id = input_id
        self.bigram_id = bigram_id
        self.field_id = field_id
        self.input_mask = input_mask
        self.seq_len = seq_len
        self.label = label


def count_lines(file_path):
    return len(codecs.open(file_path).readlines())


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    n_lines = count_lines(file_path)
    with codecs.open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm.tqdm(f, total=n_lines):
            lin = line.strip()
            if not lin:
                continue
            content_mid = lin[:lin.rfind("+")]
            content = re.sub(" +", " ", content_mid)
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1, CLS: len(vocab_dic) + 2})
    return vocab_dic


def build_dataset(config, use_word):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        with codecs.open(config.vocab_path, "r") as f:
            vocab = json.load(f)
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=config.vocab_size, min_freq=1)
        with codecs.open(config.vocab_path, "w") as f:
            json.dump(vocab, f, ensure_ascii=False)
    LOGGER.info(f"Vocab size: {len(vocab)}")

    try:
        with codecs.open(config.field_vocab, "r", encoding="utf-8") as f:
            field_vocab = json.load(f)
    except:
        field_vocab = {}

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def load_dataset(path):
        contents = []
        with codecs.open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm.tqdm(f, total=count_lines(path)):
                lin = line.strip()
                if not lin:
                    continue
                content_mid = lin[:lin.rfind("+")]
                label = lin[lin.rfind("+") + 1:]
                content = re.sub(" +", " ", content_mid)
                token_l = tokenizer(content)
                seq_len = len(token_l)
                if config.pad_size:
                    if len(token_l) < config.pad_size:
                        token_l.extend([vocab.get(PAD)] * (config.pad_size - len(token_l)))
                        mask = [1] * len(token_l) + [0] * (config.pad_size - len(token_l))
                    else:
                        token_l = token_l[:config.pad_size]
                        seq_len = config.pad_size
                        mask = [1] * config.pad_size

                # word to id
                words_line = []
                for word in token_l:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                # fasttext ngram
                buckets = config.n_gram_vocab
                bigram = []
                for i in range(config.pad_size):
                    bigram.append(biGramHash(words_line, i, buckets))

                # field feature to id
                field_words_line = []
                for word in token_l:
                    field_words_line.append(field_vocab.get(word, 0))

                feature = InputFeatures(np.array(words_line, dtype=np.int16),
                                        np.array(bigram, dtype=np.uint16),
                                        np.array(field_words_line, dtype=np.int8),
                                        np.array(mask, dtype=np.int8),
                                        seq_len,
                                        int(label))

                contents.append(feature)
        return contents
    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return train, dev, test


def build_dataset_for_test(config, use_word):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        with codecs.open(config.vocab_path, "r") as f:
            vocab = json.load(f)
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=config.vocab_size, min_freq=1)
        with codecs.open(config.vocab_path, "w") as f:
            json.dump(vocab, f, ensure_ascii=False)
    LOGGER.info(f"Vocab size: {len(vocab)}")

    try:
        with codecs.open(config.field_vocab, "r", encoding="utf-8") as f:
            field_vocab = json.load(f)
    except:
        field_vocab = {}

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def load_dataset(path):
        contents = []
        with codecs.open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm.tqdm(f, total=count_lines(path)):
                lin = line.strip()
                if not lin:
                    continue
                content_mid = lin[:lin.rfind("+")]
                label = lin[lin.rfind("+") + 1:]
                content = re.sub(" +", " ", content_mid)
                token_l = tokenizer(content)
                seq_len = len(token_l)
                if len(token_l) < config.pad_size:
                    token_l.extend([vocab.get(PAD)] * (config.pad_size - len(token_l)))
                    mask = [1] * len(token_l) + [0] * (config.pad_size - len(token_l))
                else:
                    token_l = token_l[:config.pad_size]
                    seq_len = config.pad_size
                    mask = [1] * config.pad_size

                # word to id
                words_line = []
                for word in token_l:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                # fasttext ngram
                buckets = config.n_gram_vocab
                bigram = []
                for i in range(config.pad_size):
                    bigram.append(biGramHash(words_line, i, buckets))

                # field feature to id
                field_words_line = []
                for word in token_l:
                    field_words_line.append(field_vocab.get(word, 0))

                feature = InputFeatures(np.array(words_line, dtype=np.int16),
                                        np.array(bigram, dtype=np.uint16),
                                        np.array(field_words_line, dtype=np.int8),
                                        np.array(mask, dtype=np.int8),
                                        seq_len,
                                        int(label))

                contents.append(feature)
        return contents
    test = load_dataset(config.test_path)
    return test


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
