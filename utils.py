import models
import data_loader

import numpy as np
import os
import shutil
import json
import time

import torch
import torch.optim as optim
import torch.nn as nn


# 模型保存
def save_checkpoint(state, is_best, checkpoint):
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


# 模型加载
def load_checkpoint(checkpoint, optimizer=False):
    """Loads entire model from file_path. If optimizer is True, loads
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        optimizer: (bool) resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ValueError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    if optimizer:
        return checkpoint['model'], checkpoint['optim']
    return checkpoint['model']


# 日志
def set_logger(config, s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(config.log_dir, config.log_save_name), 'a+') as f_log:
            f_log.write(s + '\n')


# 计算loss
def calculate_loss(target, predict, mask):
    loss_function = nn.CrossEntropyLoss(reduction='none')
    loss = loss_function(predict, target)
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


def to_tup(triple_list):
    ret = []
    for triple in triple_list:
        ret.append(tuple(triple))
    return ret


# 测试
def test(config, test_data_loader, model, current_f1, output=True):
    orders = ['subject', 'relation', 'object']

    id2rel = json.load(open(os.path.join(config.data_path, 'rel2id.json')))[0]
    correct_num, predict_num, gold_num = 0, 0, 0
    results = []
    test_num = 0
    s_time = time.time()
    for data in test_data_loader:
        if data is not None:
            with torch.no_grad():
                print('\r Testing step {} / {}, Please Waiting!'.format(test_num, test_data_loader.dataset.__len__()),
                      end="")
                token_ids = data['token_ids']
                tokens = data['tokens'][0]
                mask = data['mask']
                pred_triple_matrix = model(data, train=False).cpu()[0]
                rel_numbers, seq_lens, seq_lens = pred_triple_matrix.shape
                relations, subs, objs = np.where(pred_triple_matrix > 1)
                triple_list = []
                pair_numbers = len(relations)
                if pair_numbers > 0:
                    for rel, sub_start, obj_start in zip(relations, subs, objs):
                        i = sub_start + 1
                        j = obj_start + 1
                        while i < seq_lens and 0 < pred_triple_matrix[rel, i, obj_start] < 2:
                            i += 1
                        while j < seq_lens and 0 < pred_triple_matrix[rel, sub_start, j] < 2:
                            j += 1
                        sub_head, sub_tail = sub_start, i - 1
                        obj_head, obj_tail = obj_start, j - 1
                        sub = tokens[sub_head: sub_tail + 1]
                        # sub
                        sub = ''.join([i.lstrip("##") for i in sub])
                        sub = ' '.join(sub.split('[unused1]')).strip()
                        obj = tokens[obj_head: obj_tail + 1]
                        # obj
                        obj = ''.join([i.lstrip("##") for i in obj])
                        obj = ' '.join(obj.split('[unused1]')).strip()
                        rel = id2rel[str(int(rel))]
                        if len(sub) > 0 and len(obj) > 0:
                            triple_list.append((sub, rel, obj))
                triple_set = set()
                for s, r, o in triple_list:
                    triple_set.add((s, r, o))
                pred_list = list(triple_set)
                pred_triples = set(pred_list)
                gold_triples = set(to_tup(data['triples'][0]))
                correct_num += len(pred_triples & gold_triples)
                predict_num += len(pred_triples)
                gold_num += len(gold_triples)
                if output:
                    results.append({
                        'text': ' '.join(tokens[1:-1]).replace(' [unused1]', '').replace(' ##', ''),
                        'triple_list_gold': [
                            dict(zip(orders, triple)) for triple in gold_triples
                        ],
                        'triple_list_pred': [
                            dict(zip(orders, triple)) for triple in pred_triples
                        ],
                        'new': [
                            dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                        ],
                        'lack': [
                            dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                        ]
                    })
            test_num += 1
    print("\n correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    if output and f1_score > current_f1:
        if not os.path.exists(config.result_dir):
            os.makedirs(config.result_dir)
        path = os.path.join(config.result_dir, config.result_save_name)
        fw = open(path, 'w')
        for line in results:
            fw.write(json.dumps(line, ensure_ascii=False, indent=4) + "\n")
        fw.close()
    return precision, recall, f1_score


def train(config):
    # 模型搭建
    model_2DEPT = models.Model_2DEPT(config)
    if config.load_checkpoint:
        checkpoint = os.path.join(config.checkpoint_dir, config.model_save_name)
        model_2DEPT = load_checkpoint(checkpoint)
    model_2DEPT.cuda()
    # 多GPU or 单GPU:
    if config.multi_gpu:
        model = nn.DataParallel(model_2DEPT)
    else:
        model = model_2DEPT

    # 优化器准备
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_2DEPT.parameters()), lr=config.learning_rate)

    # 数据准备
    # training data
    train_data_loader = data_loader.get_loader(config, prefix=config.train_prefix, num_workers=2)
    # dev data
    test_data_loader = data_loader.get_loader(config, prefix=config.test_prefix, is_test=True)

    # check the check_point dir
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    # check the log dir
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    # other
    model.train()
    global_step = 0
    loss_sum = 0
    best_f1_score = 0
    best_precision = 0
    best_recall = 0
    best_epoch = 0
    init_time = time.time()
    start_time = time.time()

    # the training loop
    for epoch in range(config.max_epoch):
        epoch_start_time = time.time()
        for data in train_data_loader:
            if data is not None:
                pred_triple_matrix = model(data)
                triple_loss = calculate_loss(data['triple_matrix'].cuda(), pred_triple_matrix, data['loss_mask'].cuda())
                optimizer.zero_grad()
                triple_loss.backward()
                optimizer.step()
                global_step += 1
                loss_sum += triple_loss.item()
                if global_step % config.period == 0:
                    cur_loss = loss_sum / config.period
                    elapsed = time.time() - start_time
                    set_logger(config, "epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f}".
                               format(epoch, global_step, elapsed * 1000 / config.period, cur_loss * 1e6))
                    loss_sum = 0
                    start_time = time.time()

        print("total time {}".format(time.time() - epoch_start_time))

        eval_start_time = time.time()
        model.eval()
        # call the test function
        precision, recall, f1_score = test(config, test_data_loader, model, current_f1=best_f1_score,
                                           output=config.result_save_name)

        set_logger(config, 'epoch {:3d}, eval time: {:5.2f}s, f1: {:4.3f}, precision: {:4.3f}, recall: {:4.3f}'.
                   format(epoch, time.time() - eval_start_time, f1_score, precision, recall))
        improve_f1_score = f1_score - best_f1_score
        if improve_f1_score > 0:
            best_f1_score = f1_score
            best_epoch = epoch
            best_precision = precision
            best_recall = recall
            set_logger(config, "saving the model, epoch: {:3d}, precision: {:4.3f}, recall: {:4.3f}, best f1: {:4.3f}".
                       format(best_epoch, best_precision, best_recall, best_f1_score))
            # save the best model
            path = os.path.join(config.checkpoint_dir, config.model_save_name)
            torch.save(model_2DEPT.state_dict(), path)

        model.train()

        # manually release the unused cache
        torch.cuda.empty_cache()

    set_logger(config, "finish training")
    set_logger(config, "best epoch: {:3d}, precision: {:4.3f}, recall: {:4.3}, best f1: {:4.3f}, total time: {:5.2f}s".
               format(best_epoch, best_precision, best_recall, best_f1_score, time.time() - init_time))

