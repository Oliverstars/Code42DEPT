import torch
from torch import nn


class Biaffine(nn.Module):
    def __init__(self, hidden_size, tag_size, rel_num):
        super(Biaffine, self).__init__()
        self.hidden_size = hidden_size
        self.tag_size = tag_size
        self.rel_num = rel_num
        self.U1 = nn.Parameter(
            torch.FloatTensor(self.tag_size * self.rel_num, self.hidden_size + 1, self.hidden_size + 1))

        nn.init.xavier_normal_(self.U1)

    def forward(self, inputs1, inputs2):
        batch_size = inputs1.size(0)
        seq_len = inputs1.size(1)
        inputs1 = torch.cat([inputs1, torch.ones_like(inputs1[..., :1])], dim=-1)
        inputs2 = torch.cat([inputs2, torch.ones_like(inputs2[..., :1])], dim=-1)
        biaffine = torch.einsum('bxi, oij, byj -> boxy', inputs1, self.U1,
                                inputs2)  # (bs,tag_size*rel_num,seq_len,seq_len)
        return biaffine.view(batch_size, self.tag_size, self.rel_num, seq_len, seq_len)
