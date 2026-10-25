import torch
import torch.nn as nn
import torch.nn.functional as F

from timeseries.models._common import RevIN, check_input


class ExponentialDecomp(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):
        trend = [x[:, :1, :]]
        for i in range(1, x.size(1)):
            trend.append(self.alpha * x[:, i : i + 1, :] + (1.0 - self.alpha) * trend[-1])
        trend = torch.cat(trend, dim=1)
        return x - trend, trend


class DualStreamNet(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len=8, stride=4):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.patch_len = min(int(patch_len), self.seq_len)
        self.stride = max(1, min(int(stride), self.patch_len))
        self.patch_num = max(1, (self.seq_len + self.stride - self.patch_len) // self.stride + 1)
        dim = self.patch_len * self.patch_len
        self.patch_embed = nn.Sequential(nn.Linear(self.patch_len, dim), nn.GELU())
        self.depthwise = nn.Conv1d(
            self.patch_num,
            self.patch_num,
            self.patch_len,
            stride=self.patch_len,
            groups=self.patch_num,
        )
        self.res_proj = nn.Linear(dim, self.patch_len)
        self.pointwise = nn.Conv1d(self.patch_num, self.patch_num, 1)
        self.nonlinear_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.patch_len, self.pred_len * 2),
            nn.GELU(),
            nn.Linear(self.pred_len * 2, self.pred_len),
        )
        self.linear_head = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len * 4),
            nn.GELU(),
            nn.Linear(self.pred_len * 4, self.pred_len),
        )
        self.fuse = nn.Linear(self.pred_len * 2, self.pred_len)

    def forward(self, season, trend):
        b, t, d = season.shape
        s = season.transpose(1, 2).reshape(b * d, t)
        pad_len = max(0, (self.patch_num - 1) * self.stride + self.patch_len - t)
        if pad_len:
            s = F.pad(s, (0, pad_len), mode="replicate")
        p = s.unfold(-1, self.patch_len, self.stride)[:, : self.patch_num, :]
        p = self.patch_embed(p)
        res = self.res_proj(p)
        p = self.depthwise(p) + res
        p = self.pointwise(p)
        s_out = self.nonlinear_head(p)

        t_out = self.linear_head(trend.transpose(1, 2).reshape(b * d, t))
        y = self.fuse(torch.cat([s_out, t_out], dim=-1)).reshape(b, d, self.pred_len)
        return y.transpose(1, 2)


class Model(nn.Module):
    """xPatch core: exponential decomposition plus nonlinear patch and linear streams."""

    def __init__(self, configs=None, seq_len=16, pred_len=1, d=5):
        super().__init__()
        if configs is not None and hasattr(configs, 'seq_len'):
            seq_len, pred_len, d = configs.seq_len, configs.pred_len, configs.enc_in
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d = int(d)
        self.revin = RevIN(self.d)
        self.decomp = ExponentialDecomp(alpha=0.3)
        self.net = DualStreamNet(self.seq_len, self.pred_len)

    def forward(self, x, x_mark=None, dec_inp=None, y_mark=None, mask=None):
        check_input(x, self.seq_len, self.d)
        x = self.revin.norm(x)
        season, trend = self.decomp(x)
        y = self.net(season, trend)
        return self.revin.denorm(y)
