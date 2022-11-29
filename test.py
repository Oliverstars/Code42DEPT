import config
import argparse
import models
import os
import torch
import numpy as np
import random
import data_loader
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
parser.add_argument('--dataset', type=str, default='WebNLG')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--train_prefix', type=str, default='train_triples')
parser.add_argument('--dev_prefix', type=str, default='dev_triples')
parser.add_argument('--test_prefix', type=str, default='test_triples')
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--bert_max_len', type=int, default=200)
parser.add_argument('--rel_num', type=int, default=216)
parser.add_argument('--period', type=int, default=100)
parser.add_argument('--load_checkpoint', type=bool, default=False)
args = parser.parse_args()
# 模型配置
model_config = config.Config(args)
model = models.Model_2DEPT(model_config)
path = os.path.join(model_config.checkpoint_dir, model_config.model_save_name)
model.load_state_dict(torch.load(path))
model.cuda()
model.eval()
test_data_loader = data_loader.get_loader(model_config, prefix=model_config.test_prefix, is_test=True)
precision, recall, f1_score = test(model_config,test_data_loader, model, current_f1=0, output=True)
print("f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}".format(f1_score, precision, recall))


