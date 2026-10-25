import torch
import torch.nn as nn
import torch.nn.functional as F

from timeseries.models._common import check_input


class Model(nn.Module):
    """LSTNet-style CNN + GRU + skip-GRU + highway forecaster."""

    def __init__(self, configs=None, seq_len=16, pred_len=1, d=5):
        super().__init__()
        if configs is not None and hasattr(configs, 'seq_len'):
            seq_len, pred_len, d = configs.seq_len, configs.pred_len, configs.enc_in
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d = int(d)
        hid_cnn = 64
        hid_rnn = 64
        hid_skip = 16
        cnn_kernel = min(6, self.seq_len)
        self.skip = 4 if self.seq_len >= 8 else 0
        self.pt = max(1, (self.seq_len - cnn_kernel) // max(1, self.skip)) if self.skip else 0
        self.highway_window = min(8, self.seq_len)

        self.conv = nn.Conv2d(1, hid_cnn, kernel_size=(cnn_kernel, self.d))
        self.gru = nn.GRU(hid_cnn, hid_rnn)
        self.skip_gru = nn.GRU(hid_cnn, hid_skip) if self.skip else None
        linear_in = hid_rnn + self.skip * hid_skip if self.skip else hid_rnn
        self.main_head = nn.Linear(linear_in, self.pred_len * self.d)
        self.highway = nn.Linear(self.highway_window, self.pred_len)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, x_mark=None, dec_inp=None, y_mark=None, mask=None):
        check_input(x, self.seq_len, self.d)
        b = x.size(0)
        c = F.relu(self.conv(x.view(b, 1, self.seq_len, self.d))).squeeze(3)
        c = self.dropout(c)
        r_in = c.permute(2, 0, 1).contiguous()
        _, r = self.gru(r_in)
        r = self.dropout(r.squeeze(0))

        if self.skip:
            s = c[:, :, -self.pt * self.skip :].contiguous()
            s = s.view(b, c.size(1), self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous().view(self.pt, b * self.skip, c.size(1))
            _, s = self.skip_gru(s)
            s = self.dropout(s.view(b, self.skip * s.size(-1)))
            r = torch.cat([r, s], dim=1)

        y = self.main_head(r).view(b, self.pred_len, self.d)
        hw = x[:, -self.highway_window :, :].transpose(1, 2)
        y = y + self.highway(hw).transpose(1, 2)
        return y
