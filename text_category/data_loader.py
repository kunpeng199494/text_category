# -*- coding: utf-8 -*-

import torch
from text_category.utils import build_dataset, build_dataset_for_test
from torch.utils.data import DataLoader, TensorDataset


def data_loader(config):
    train_data, dev_data, test_data = build_dataset(config, use_word=config.use_word)

    train_ids = torch.LongTensor([temp.input_id for temp in train_data]).to(config.device)
    train_bigram = torch.LongTensor([temp.bigram_id for temp in train_data]).to(config.device)
    train_field = torch.LongTensor([temp.field_id for temp in train_data]).to(config.device)
    train_masks = torch.LongTensor([temp.input_mask for temp in train_data]).to(config.device)
    train_seq = torch.LongTensor([temp.seq_len for temp in train_data]).to(config.device)
    train_label = torch.LongTensor([temp.label for temp in train_data]).to(config.device)

    train_dataset = TensorDataset(train_ids, train_bigram, train_field, train_masks, train_seq, train_label)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

    dev_ids = torch.LongTensor([temp.input_id for temp in dev_data]).to(config.device)
    dev_bigram = torch.LongTensor([temp.bigram_id for temp in dev_data]).to(config.device)
    dev_field = torch.LongTensor([temp.field_id for temp in dev_data]).to(config.device)
    dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data]).to(config.device)
    dev_seq = torch.LongTensor([temp.seq_len for temp in dev_data]).to(config.device)
    dev_label = torch.LongTensor([temp.label for temp in dev_data]).to(config.device)

    dev_dataset = TensorDataset(dev_ids, dev_bigram, dev_field, dev_masks, dev_seq, dev_label)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)

    test_ids = torch.LongTensor([temp.input_id for temp in test_data]).to(config.device)
    test_bigram = torch.LongTensor([temp.bigram_id for temp in test_data]).to(config.device)
    test_field = torch.LongTensor([temp.field_id for temp in test_data]).to(config.device)
    test_masks = torch.LongTensor([temp.input_mask for temp in test_data]).to(config.device)
    test_seq = torch.LongTensor([temp.seq_len for temp in test_data]).to(config.device)
    test_label = torch.LongTensor([temp.label for temp in test_data]).to(config.device)

    test_dataset = TensorDataset(test_ids, test_bigram, test_field, test_masks, test_seq, test_label)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)

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
