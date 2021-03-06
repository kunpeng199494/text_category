# -*- coding: utf-8 -*-

import torch
from text_category.utils import build_dataset, build_dataset_for_test
from torch.utils.data import DataLoader, TensorDataset


def data_loader_base(temp_data, config):
    temp_ids = torch.LongTensor([temp.input_id for temp in temp_data]).to(config.device)
    temp_bigram = torch.LongTensor([temp.bigram_id for temp in temp_data]).to(config.device)
    temp_field = torch.LongTensor([temp.field_id for temp in temp_data]).to(config.device)
    temp_masks = torch.LongTensor([temp.input_mask for temp in temp_data]).to(config.device)
    temp_seq = torch.LongTensor([temp.seq_len for temp in temp_data]).to(config.device)
    temp_label = torch.LongTensor([temp.label for temp in temp_data]).to(config.device)
    temp_dataset = TensorDataset(temp_ids, temp_bigram, temp_field, temp_masks, temp_seq, temp_label)
    temp_loader = DataLoader(temp_dataset, shuffle=True, batch_size=config.batch_size)
    return temp_loader


def data_loader(config):
    train_data, dev_data, test_data = build_dataset(config, use_word=config.use_word)

    train_loader = data_loader_base(train_data, config)
    dev_loader = data_loader_base(dev_data, config)
    test_loader = data_loader_base(test_data, config)

    return train_loader, dev_loader, test_loader


def batch_predict_loader(config):
    test_data = build_dataset_for_test(config, use_word=config.use_word)

    test_ids = torch.LongTensor([temp.input_id for temp in test_data]).to(config.device)
    test_bigram = torch.LongTensor([temp.bigram_id for temp in test_data]).to(config.device)
    test_field = torch.LongTensor([temp.field_id for temp in test_data]).to(config.device)
    test_masks = torch.LongTensor([temp.input_mask for temp in test_data]).to(config.device)
    test_seq = torch.LongTensor([temp.seq_len for temp in test_data]).to(config.device)
    test_label = torch.LongTensor([temp.label for temp in test_data]).to(config.device)

    test_dataset = TensorDataset(test_ids, test_bigram, test_field, test_masks, test_seq, test_label)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)

    return test_loader
