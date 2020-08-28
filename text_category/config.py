# -*- coding: utf-8 -*-

import os
import glob
import torch
import hao

root_dir = hao.paths.project_root_path()


class Config(object):

    def __init__(self, model_name, dataset):
        # General
        self.model_name = model_name
        # 数据相关的根目录
        self.dataset = dataset

        # 训练数据集
        self.train_path = os.path.join(self.dataset, "train_dev_test/train.txt")
        self.dev_path = os.path.join(self.dataset, "train_dev_test/dev.txt")
        self.test_path = os.path.join(self.dataset, "train_dev_test/test.txt")

        # 类别及其对应关系
        self.class_path = os.path.join(self.dataset, "classes/class.txt")
        self.class_list = [x.strip().split("+")[0] for x in open(self.class_path, encoding="utf-8").readlines()]

        # 领域词表，格式为一个领域词表为一个txt文件，该文件每行为一个词汇
        self.field_vocab = os.path.join(self.dataset, "field")
        # 领域词表个数
        self.field_size = len(glob.glob(self.field_vocab + "/*.txt")) + 1

        # 预训练BERT模型
        self.bert_path = os.path.join(self.dataset, "bert_path")

        # 模型保存
        self.save_path = os.path.join(self.dataset, f"saved_dict/{self.model_name}.ckpt")
        # 字典
        self.vocab_path = os.path.join(self.dataset, "vocab/vocab.json")

        os.makedirs(os.path.split(self.train_path)[0], exist_ok=True)
        os.makedirs(os.path.split(self.class_path)[0], exist_ok=True)
        os.makedirs(os.path.split(self.save_path)[0], exist_ok=True)
        os.makedirs(os.path.split(self.bert_path)[0], exist_ok=True)
        os.makedirs(os.path.split(self.vocab_path)[0], exist_ok=True)

        # 一些基础配置
        self.use_cuda = True
        self.device = torch.device("cuda") if self.use_cuda and torch.cuda.is_available() else torch.device("cpu")
        self.gpu = 0
        self.map_location = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.require_improvement = 1000
        self.classes = len(self.class_list)
        self.epochs = 50
        self.batch_size = 32
        self.pad_size = 128
        self.vocab_size = 10000
        self.warmup_ratio = 0.1
        self.embedding_pretrained = None

        if model_name.startswith("BERT"):
            self.learning_rate = 5e-5
        elif model_name.startswith("Transformer"):
            self.learning_rate = 5e-4
        else:
            self.learning_rate = 1e-3

        # 按照词汇的方式进行分类，如果为False，则按照字符的方式进行分类
        self.use_word = True

        self.embed = 300
        self.n_gram_vocab = 25000

        self.fc_hidden_size = 1024
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.dropout = 0.1 if model_name.startswith("BERT") else 0.5
        self.weight_decay = 0.00005

        # 用于控制是否输出模型预测出正确结果时的概率
        self.prob = True
        # 概率保存位置
        if self.prob:
            self.threshold_reference_path = os.path.join(self.dataset, "threshold/threshold.json")
            os.makedirs(os.path.split(self.threshold_reference_path)[0], exist_ok=True)

        # RNN
        self.rnn_hidden = 256
        self.num_layers = 2
        self.hidden_size_rnn = 64

        # BERT
        if model_name.startswith("BERT"):
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_hidden_size = 768
        # 是否冻结BERT层
        self.bert_frozen = True

        # Transformer
        self.dim_model = 300
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
