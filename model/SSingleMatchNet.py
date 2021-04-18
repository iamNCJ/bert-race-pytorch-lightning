import torch
from torch import nn
from torch.autograd import Variable


def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i, :, :seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)

    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result


class SSingleMatchNet(nn.Module):
    def __init__(self, config):
        super(SSingleMatchNet, self).__init__()
        # self.map_linear = nn.Linear(2*config.hidden_size, 2*config.hidden_size)
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        # self.drop_module = nn.Dropout(2*config.hidden_dropout_prob)
        # self.rank_module = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2) )
        att_norm = masked_softmax(att_weights, seq_len)

        att_vec = att_norm.bmm(proj_q)
        output = nn.ReLU()(self.trans_linear(att_vec))
        return output
