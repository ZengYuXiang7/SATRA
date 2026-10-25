import torch
from torch import nn

from .model import Model as MasterModel


class MasterBatchAdapter(nn.Module):
    """Build MASTER's concat input from explicit stock and market tensors."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        if isinstance(batch, dict):
            if "stock_inputs" in batch and "market_inputs" in batch:
                stock_inputs = batch["stock_inputs"]
                market_inputs = batch["market_inputs"]
                if stock_inputs.dim() != 4:
                    raise ValueError(f"stock_inputs must be [B,N,L,F], got {tuple(stock_inputs.shape)}")
                if market_inputs.dim() != 3:
                    raise ValueError(f"market_inputs must be [B,L,Fm], got {tuple(market_inputs.shape)}")
                if stock_inputs.size(0) != market_inputs.size(0) or stock_inputs.size(2) != market_inputs.size(1):
                    raise ValueError(
                        "stock_inputs and market_inputs disagree on batch or lookback: "
                        f"{tuple(stock_inputs.shape)} vs {tuple(market_inputs.shape)}"
                    )
                market_expanded = market_inputs.unsqueeze(1).expand(
                    -1, stock_inputs.size(1), -1, -1
                )
                model_input = torch.cat([stock_inputs, market_expanded], dim=-1)
            elif "inputs" in batch:
                model_input = batch["inputs"]
            else:
                raise ValueError("MASTER batch must contain stock_inputs+market_inputs or inputs")
            stock_mask = batch.get("mask")
            if stock_mask is not None:
                stock_mask = stock_mask.squeeze(-1)
        else:
            model_input = batch
            stock_mask = None

        pred = self.model(model_input, stock_mask=stock_mask)
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        elif pred.dim() == 2:
            pred = pred.unsqueeze(-1)
        else:
            raise ValueError(f"MASTER prediction must be [N] or [B,N], got {tuple(pred.shape)}")
        return pred


def build_model(args, data):
    stock_feature_dim = int(data.stock_feature_num)
    market_feature_dim = int(data.market_feature_num)
    gate_start = stock_feature_dim
    gate_end = stock_feature_dim + market_feature_dim
    model = MasterModel(
        d_feat=stock_feature_dim,
        d_model=int(getattr(args, "d_model", 256)),
        t_nhead=int(getattr(args, "t_nhead", 4)),
        s_nhead=int(getattr(args, "s_nhead", 2)),
        t_dropout_rate=float(getattr(args, "dropout", 0.5)),
        s_dropout_rate=float(getattr(args, "dropout", 0.5)),
        gate_input_start_index=gate_start,
        gate_input_end_index=gate_end,
        beta=float(getattr(args, "gate_beta", 5.0)),
        max_len=max(512, int(args.seq_len)),
    )
    args.stock_num = int(data.stock_num)
    args.master_stock_feature_dim = stock_feature_dim
    args.master_market_feature_dim = market_feature_dim
    args.master_gate_input_start_index = gate_start
    args.master_gate_input_end_index = gate_end
    return MasterBatchAdapter(model)
