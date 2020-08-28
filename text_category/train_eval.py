# -*- coding:utf-8 -*-

import os
import json
import time
import torch
import hao
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from text_category.utils import get_time_dif
from transformers.optimization import get_linear_schedule_with_warmup
try:
    from apex.fp16_utils import FP16_Optimizer
    switch = True
except:
    switch = False

LOGGER = hao.logs.get_logger(__name__)


def init_network(model, method='kaiming', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def init_optimizer(config, model, optimizer):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optimizer(optimizer_grouped_parameters, config.learning_rate)
    return optimizer


def init_schedule(config, optimizer, train_loader):
    t_total = len(train_loader) * config.epochs
    warmup_steps = t_total * config.warmup_ratio
    if switch:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        scheduler = get_linear_schedule_with_warmup(optimizer.optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
    return scheduler


def train(config, model, train_loader, dev_loader, test_loader):
    start_time = time.time()
    model.train()

    optimizer = torch.optim.Adam
    optimizer = init_optimizer(config, model, optimizer)
    scheduler = init_schedule(config, optimizer, train_loader)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    for epoch in range(config.epochs):
        LOGGER.info('Epoch [{}/{}]'.format(epoch + 1, config.epochs))
        for i, batchs in enumerate(train_loader):
            outputs = model(batchs)
            loss = F.cross_entropy(outputs, batchs[5])
            optimizer.zero_grad()
            if switch:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            scheduler.step()
            if total_batch % 100 == 0:
                true = batchs[5].data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                LOGGER.info(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                LOGGER.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_loader)


def test(config, model, test_loader):
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_loader, test_mode=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    LOGGER.info(msg.format(test_loss, test_acc))
    LOGGER.info("Precision, Recall and F1-Score...")
    LOGGER.info(test_report)
    LOGGER.info("Confusion Matrix...")
    LOGGER.info(test_confusion)


def evaluate(config, model, data_loader, test_mode=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    probability_all = np.array([], dtype=int)
    threshold_reference = {}

    with torch.no_grad():
        for batchs in data_loader:
            outputs = model(batchs)
            loss = F.cross_entropy(outputs, batchs[5])
            loss_total += loss
            labels = batchs[5].data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            if config.prob:
                softmax_ = F.softmax(outputs.data, dim=1)
                prob_ = torch.max(softmax_, dim=1)[0].cpu().numpy()
                probability_all = np.append(probability_all, prob_)

    if config.prob:
        for i in range(len(labels_all)):
            if labels_all[i] == predict_all[i] and str(labels_all[i]) not in threshold_reference.keys():
                threshold_reference[str(labels_all[i])] = [probability_all[i]]
            elif labels_all[i] == predict_all[i] and str(labels_all[i]) in threshold_reference.keys():
                threshold_reference[str(labels_all[i])].append(probability_all[i])
            else:
                continue
        with open(config.threshold_reference_path, "w", encoding="utf-8") as f:
            json.dump(threshold_reference, f, ensure_ascii=False)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test_mode:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_loader), report, confusion

    model.train()
    return acc, loss_total / len(data_loader)
