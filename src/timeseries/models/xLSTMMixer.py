import torch
import torch.nn as nn

from timeseries.models._common import MixerBlock, RevIN, check_input


class ScalarMemoryBlock(nn.Module):
    """Small xLSTM-inspired gated recurrent block using scalar memory per hidden unit."""

    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 4)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        b, t, h = x.shape
        m = x.new_zeros(b, h)
        outs = []
        for i in range(t):
            i_gate, f_gate, o_gate, cand = self.in_proj(x[:, i]).chunk(4, dim=-1)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            o_gate = torch.sigmoid(o_gate)
            m = f_gate * m + i_gate * torch.tanh(cand)
            outs.append(o_gate * torch.tanh(m))
        return self.out_norm(torch.stack(outs, dim=1))


class Model(nn.Module):
    """xLSTM-Mixer core: scalar-memory recurrent mixing followed by MLP mixing."""

    def __init__(self, configs=None, seq_len=16, pred_len=1, d=5):
        super().__init__()
        if configs is not None and hasattr(configs, 'seq_len'):
            seq_len, pred_len, d = configs.seq_len, configs.pred_len, configs.enc_in
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d = int(d)
        d_model = 128
        self.revin = RevIN(self.d)
        self.input_proj = nn.Linear(self.d, d_model)
        self.memory = ScalarMemoryBlock(d_model)
        self.mixers = nn.ModuleList([MixerBlock(self.seq_len, d_model, hidden=256, dropout=0.1) for _ in range(2)])
        self.time_head = nn.Linear(self.seq_len, self.pred_len)
        self.output_proj = nn.Linear(d_model, self.d)

    def forward(self, x, x_mark=None, dec_inp=None, y_mark=None, mask=None):
        check_input(x, self.seq_len, self.d)
        x = self.revin.norm(x)
        z = self.input_proj(x)
        z = self.memory(z)
        for mixer in self.mixers:
            z = mixer(z)
        z = self.time_head(z.transpose(1, 2)).transpose(1, 2)
        y = self.output_proj(z)
        return self.revin.denorm(y)
