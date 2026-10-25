import torch.nn as nn

from timeseries.models._common import PatchFlattenHead, RevIN, check_input


class ModernTCNBlock(nn.Module):
    def __init__(self, d, hidden=64, large_kernel=15, small_kernel=5, dropout=0.1):
        super().__init__()
        padding_large = large_kernel // 2
        padding_small = small_kernel // 2
        self.dw_large = nn.Conv1d(d * hidden, d * hidden, large_kernel, padding=padding_large, groups=d * hidden)
        self.dw_small = nn.Conv1d(d * hidden, d * hidden, small_kernel, padding=padding_small, groups=d * hidden)
        self.norm = nn.BatchNorm1d(d * hidden)
        self.ffn_channel = nn.Sequential(
            nn.Conv1d(d * hidden, d * hidden * 2, 1, groups=d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d * hidden * 2, d * hidden, 1, groups=d),
        )
        self.ffn_hidden = nn.Sequential(
            nn.Conv1d(d * hidden, d * hidden * 2, 1, groups=hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d * hidden * 2, d * hidden, 1, groups=hidden),
        )

    def forward(self, x):
        b, d, h, n = x.shape
        y = x.reshape(b, d * h, n)
        y = self.dw_large(y) + self.dw_small(y)
        y = self.norm(y)
        y = self.ffn_channel(y) + y
        y = self.ffn_hidden(y) + y
        return x + y.reshape(b, d, h, n)


class Model(nn.Module):
    """ModernTCN core: patch stem, large-kernel depthwise conv, grouped FFNs."""

    def __init__(self, configs=None, seq_len=16, pred_len=1, d=5):
        super().__init__()
        if configs is not None and hasattr(configs, 'seq_len'):
            seq_len, pred_len, d = configs.seq_len, configs.pred_len, configs.enc_in
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d = int(d)
        self.patch_size = min(8, self.seq_len)
        self.patch_stride = max(1, self.patch_size // 2)
        hidden = 64
        patch_num = (self.seq_len + self.patch_stride - self.patch_size) // self.patch_stride + 1
        self.revin = RevIN(self.d)
        self.stem = nn.Conv1d(1, hidden, self.patch_size, stride=self.patch_stride)
        self.blocks = nn.ModuleList([ModernTCNBlock(self.d, hidden) for _ in range(2)])
        self.head = PatchFlattenHead(patch_num, hidden, self.pred_len)

    def forward(self, x, x_mark=None, dec_inp=None, y_mark=None, mask=None):
        check_input(x, self.seq_len, self.d)
        b = x.size(0)
        x = self.revin.norm(x)
        z = x.transpose(1, 2).reshape(b * self.d, 1, self.seq_len)
        pad_len = max(0, self.patch_size - self.patch_stride)
        if pad_len:
            z = nn.functional.pad(z, (0, pad_len), mode="replicate")
        z = self.stem(z)
        z = z.reshape(b, self.d, z.size(1), z.size(2))
        for block in self.blocks:
            z = block(z)
        y = self.head(z)
        return self.revin.denorm(y)
