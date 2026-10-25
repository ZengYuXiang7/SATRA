import torch
import torch.nn as nn

from timeseries.models._common import RevIN, SimpleTransformer, check_input


class TemporalExpert(nn.Module):
    def __init__(self, seq_len, pred_len, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, x):
        return self.net(x)


class SparseTemporalCluster(nn.Module):
    def __init__(self, seq_len, pred_len, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.experts = nn.ModuleList([TemporalExpert(seq_len, pred_len) for _ in range(self.num_experts)])
        self.router = nn.Sequential(nn.Linear(seq_len, 64), nn.ReLU(), nn.Linear(64, self.num_experts))

    def forward(self, x):
        gates = self.router(x)
        top_val, top_idx = torch.topk(gates, k=self.top_k, dim=-1)
        weights = torch.softmax(top_val, dim=-1)
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=-1)
        selected = torch.gather(expert_out, -1, top_idx.unsqueeze(1).expand(-1, expert_out.size(1), -1))
        return (selected * weights.unsqueeze(1)).sum(dim=-1), gates


class Model(nn.Module):
    """DUET core: temporal mixture-of-experts plus channel Transformer clustering."""

    def __init__(self, configs=None, seq_len=16, pred_len=1, d=5):
        super().__init__()
        if configs is not None and hasattr(configs, 'seq_len'):
            seq_len, pred_len, d = configs.seq_len, configs.pred_len, configs.enc_in
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d = int(d)
        d_model = 128
        self.revin = RevIN(self.d)
        self.temporal_cluster = SparseTemporalCluster(self.seq_len, d_model, num_experts=4, top_k=2)
        self.channel_encoder = SimpleTransformer(d_model=d_model, n_heads=4, layers=2, d_ff=256, dropout=0.1)
        self.head = nn.Linear(d_model, self.pred_len)

    def forward(self, x, x_mark=None, dec_inp=None, y_mark=None, mask=None):
        check_input(x, self.seq_len, self.d)
        x = self.revin.norm(x)
        b = x.size(0)
        z = x.transpose(1, 2).reshape(b * self.d, self.seq_len)
        z, _ = self.temporal_cluster(z)
        z = z.reshape(b, self.d, -1)
        z = self.channel_encoder(z)
        y = self.head(z).transpose(1, 2)
        return self.revin.denorm(y)
