import config
from utils import *
import argparse
import os
import torch
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set random seed
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='2DEPT', help='name of the model')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--dropout_prob', type=float, default=0.2)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='NYT')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--train_prefix', type=str, default='train_triples')
parser.add_argument('--dev_prefix', type=str, default='dev_triples')
parser.add_argument('--test_prefix', type=str, default='test_triples')
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--bert_max_len', type=int, default=200)
parser.add_argument('--rel_num', type=int, default=24)
parser.add_argument('--period', type=int, default=500)
parser.add_argument('--load_checkpoint', type=bool, default=False)
args = parser.parse_args()
# 模型配置
model_config = config.Config(args)

train(model_config)
