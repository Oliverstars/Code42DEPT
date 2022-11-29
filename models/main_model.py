from torch import nn
from transformers import BertModel

from .biaffine import Biaffine


class Model_2DEPT(nn.Module):
    def __init__(self, config):
        super(Model_2DEPT, self).__init__()
        self.config = config
        self.bert_dim = config.bert_dim
        self.bert_encoder = BertModel.from_pretrained("bert-base-cased", cache_dir='./pre_trained_bert')
        self.dropout = nn.Dropout(config.dropout_prob)
        self.sub = nn.Linear(self.bert_dim, 3*self.bert_dim)
        self.obj = nn.Linear(self.bert_dim, 3*self.bert_dim)
        self.init_weight()

        self.biaffine = Biaffine(3*self.bert_dim, self.config.tag_size, self.config.rel_num)

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]  # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.dropout(encoded_text)
        return encoded_text

    def init_weight(self):
        nn.init.xavier_uniform_(self.sub.weight)
        nn.init.xavier_uniform_(self.obj.weight)

    def forward(self, data, train=True):
        token_ids = data['token_ids'].cuda()  # [batch_size, seq_len]
        mask = data['mask'].cuda()  # [batch_size, seq_len]
        encoded_text = self.get_encoded_text(token_ids, mask)  # [batch_size, seq_len, bert_dim(768)]
        subs = self.sub(encoded_text)
        objs = self.obj(encoded_text)
        output = self.biaffine(subs, objs)  # [batch_size, tag_size, rel_num, seq_len, seq_len]
        if train:
            return output
        return output.argmax(dim=1)


if __name__ == '__main__':
    pass
