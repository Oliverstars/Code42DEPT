import os


class Config(object):
    def __init__(self, args):
        self.multi_gpu = args.multi_gpu
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.test_epoch = args.test_epoch
        self.max_len = args.max_len
        self.rel_num = args.rel_num
        self.bert_max_len = args.bert_max_len
        self.bert_dim = 768
        self.tag_size = 3
        self.dropout_prob = args.dropout_prob
        self.period = args.period
        self.load_checkpoint=args.load_checkpoint

        # dataset
        self.dataset = args.dataset

        # path and name
        self.root_path = '/'.join(os.path.dirname(__file__).split('/')[:-1])
        self.data_path = self.root_path+'/data/'+self.dataset
        self.checkpoint_dir = self.root_path+'/checkpoint/'+self.dataset
        self.log_dir = self.root_path+'/log/'+self.dataset
        self.result_dir = self.root_path+'/result/'+self.dataset

        self.train_prefix = args.train_prefix
        self.dev_prefix = args.dev_prefix
        self.test_prefix = args.test_prefix

        self.model_save_name = args.model_name + '_' + self.dataset + '_' + str(self.rel_num) + '_best.pth'
        self.log_save_name = 'LOG_' + args.model_name + '_' + self.dataset + '_' + str(self.rel_num) + '.log'
        self.result_save_name = 'RESULT_' + args.model_name + '_' + self.dataset + '_' + str(
            self.rel_num) + '_result.json'

        self.min_epoch_num = 200
        self.patience = 0.000001
        self.patience_num = 500


if __name__ == '__main__':
    import argparse

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
    parser.add_argument('--period', type=int, default=100)
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    args = parser.parse_args()
    my_config = Config(args)
    print(my_config)
