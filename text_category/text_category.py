# -*- coding: utf-8 -*-

import os
import time
import json
import codecs
import torch
import numpy as np
from text_category import models
from torch.utils.data import TensorDataset
from text_category.data_loader import data_loader, batch_predict_loader
from text_category.train_eval import train, init_network, test
from text_category.utils import get_time_dif, InputFeatures


class Train(object):
    def __init__(self, model_name, config):
        import hao
        self.LOGGER = hao.logs.get_logger(__name__)
        self.model_name = model_name
        self.config = config
        self.train_process()

    def train_process(self):
        if self.config.use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)
        model = getattr(models, self.model_name)

        start_time = time.time()
        self.LOGGER.info("Loading data...")

        train_loader, dev_loader, test_loader = data_loader(self.config)

        vocab_size = self.config.vocab_size
        mid = vocab_size + 3
        self.config.update(vocab_size=mid)

        model = model(self.config).to(self.config.device)
        if self.model_name != 'Transformer' and not self.model_name.startswith("BERT"):
            init_network(model)

        self.LOGGER.info(model.parameters)
        train(self.config, model, train_loader, dev_loader, test_loader)

        time_dif = get_time_dif(start_time)
        self.LOGGER.info(f"Time usage: {time_dif}")


class Test(object):
    def __init__(self, model_name, config):
        import hao
        self.LOGGER = hao.logs.get_logger(__name__)
        self.model_name = model_name
        self.config = config
        self.test_process()

    def test_process(self):
        if self.config.use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)
        model = getattr(models, self.model_name)

        vocab_size = self.config.vocab_size
        mid = vocab_size + 3
        self.config.update(vocab_size=mid)

        start_time = time.time()
        self.LOGGER.info("Loading data...")

        test_loader = batch_predict_loader(self.config)

        model = model(self.config).to(self.config.device)
        model.load_state_dict(torch.load(self.config.save_path, map_location=self.config.map_location))
        test(self.config, model, test_loader)

        time_dif = get_time_dif(start_time)
        self.LOGGER.info(f"Time usage: {time_dif}")


class Predict(object):
    def __init__(self, model_name, config):
        import hao
        self.LOGGER = hao.logs.get_logger(__name__)
        self.model_name = model_name
        self.config = config

    @staticmethod
    def mapping(config):
        index_dict = {}
        category_true = []
        category_fake = []
        with open(config.class_path, 'r') as f:
            for line in f.readlines():
                category_true.append(line.strip().split("+")[0].strip())
                category_fake.append(line.strip().split("+")[1].strip())
        for i in range(len(category_true)):
            index_dict[str(int(category_fake[i]))] = str(category_true[i])
        return index_dict

    # 输入的line是分过词的句子
    def predict_line(self, line: str):
        if self.config.use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)
        model = getattr(models, self.model_name)
        start_time = time.time()
        index_dict = self.mapping(self.config)

        vocab_size = self.config.vocab_size
        mid = vocab_size + 3
        self.config.update(vocab_size=mid)

        vocab = process_cls.load_vocab(self.config)
        contents = process_cls.load_data(vocab, line, self.config)
        data = process_cls.data_to_tensor(self.config, contents)
        model = model(self.config).to(self.config.device)

        model.load_state_dict(torch.load(self.config.save_path, map_location=self.config.map_location))
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            fake = torch.max(outputs.data, 1)[1].cpu().numpy()[0]

        time_dif = get_time_dif(start_time)
        self.LOGGER.info(f"Time usage:{time_dif}")
        return str(index_dict[str(int(fake))])

    # 输入的data_list是分过词的句子的列表
    def predict_list(self, data_list: list):
        if self.config.use_cuda:
            torch.cuda.set_device(self.config.gpu)
        model = getattr(models, self.model_name)
        start_time = time.time()
        index_dict = self.mapping(self.config)
        vocab = process_cls.load_vocab(self.config)
        contents = [process_cls.load_data(vocab, line, self.config) for line in data_list]
        data = [process_cls.data_to_tensor(self.config, content) for content in contents]
        model = model(config).to(config.device)
        model.load_state_dict(torch.load(config.save_path, map_location=self.config.map_location))
        model.eval()
        with torch.no_grad():
            outputs = [model(element) for element in data]
            fake = [torch.max(output.data, 1)[1].cpu().numpy()[0] for output in outputs]

        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        return [str(index_dict[str(int(fake_))]) for fake_ in fake]


class process_cls(object):

    @staticmethod
    def biGramHash(sequence, t_m, buckets):
        t1 = sequence[t_m - 1] if t_m - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    @staticmethod
    def load_vocab(config):
        with codecs.open(config.vocab_path, "r") as f:
            vocab = json.load(f)
        return vocab

    @staticmethod
    def load_data(vocab, line, config):
        UNK, PAD, CLS = '[UNK]', '[PAD]', '[CLS]'
        tokenizer = lambda x: x.split(' ')
        contents = []
        content = line.strip()
        token_l = tokenizer(content)
        seq_len = len(token_l)
        if len(token_l) < config.pad_size:
            token_l.extend([vocab.get(PAD)] * (config.pad_size - len(token_l)))
            mask = [1] * len(token_l) + [0] * (config.pad_size - len(token_l))
        else:
            token_l = token_l[:pad_size]
            seq_len = config.pad_size
            mask = [1] * config.pad_size
        # word to id
        words_line = []
        for word in token_l:
            words_line.append(vocab.get(word, vocab.get(UNK)))

        buckets = 25000
        bigram = []
        for i in range(config.pad_size):
            bigram.append(process_cls.biGramHash(words_line, i, buckets))

        # field feature to id
        try:
            with codecs.open(config.field_vocab, "r", encoding="utf-8") as f:
                field_vocab = json.load(f)
        except:
            field_vocab = {}

        field_words_line = []
        for word in token_l:
            field_words_line.append(field_vocab.get(word, 0))

        feature = InputFeatures(np.array(words_line, dtype=np.int16),
                                np.array(bigram, dtype=np.uint16),
                                np.array(field_words_line, dtype=np.int8),
                                np.array(mask, dtype=np.int8),
                                seq_len,
                                None)
        contents.append(feature)
        return contents

    @staticmethod
    def data_to_tensor(config, test_data):
        test_ids = torch.LongTensor([temp.input_id for temp in test_data]).to(config.device)
        test_bigram = torch.LongTensor([temp.bigram_id for temp in test_data]).to(config.device)
        test_field = torch.LongTensor([temp.field_id for temp in test_data]).to(config.device)
        test_masks = torch.LongTensor([temp.input_mask for temp in test_data]).to(config.device)
        test_seq = torch.LongTensor([temp.seq_len for temp in test_data]).to(config.device)

        test_dataset = [test_ids, test_bigram, test_field, test_masks, test_seq]
        return test_dataset
