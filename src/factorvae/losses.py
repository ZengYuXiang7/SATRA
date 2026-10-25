import torch
import torch.nn.functional as F


def _ensure_batched(tensor):
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 3:
        return tensor
    raise ValueError(f"expected [stocks,1] or [batch,stocks,1], got {tuple(tensor.shape)}")


def return_rank_loss(prediction, ground_truth, mask, stock_num, alpha=0.1):
    prediction = _ensure_batched(prediction)
    ground_truth = _ensure_batched(ground_truth)
    mask = _ensure_batched(mask)
    if prediction.shape != ground_truth.shape or prediction.shape != mask.shape:
        raise ValueError(
            "prediction, ground_truth and mask must share shape, got "
            f"{tuple(prediction.shape)}, {tuple(ground_truth.shape)}, {tuple(mask.shape)}"
        )
    if prediction.shape[1] != int(stock_num):
        raise ValueError(f"stock dimension mismatch: got {prediction.shape[1]}, expected {stock_num}")

    reg_loss = F.mse_loss(prediction * mask, ground_truth * mask)
    pred = prediction.squeeze(-1)
    gt = ground_truth.squeeze(-1)
    valid = mask.squeeze(-1)
    pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
    gt_diff = gt.unsqueeze(1) - gt.unsqueeze(2)
    pair_mask = valid.unsqueeze(2) * valid.unsqueeze(1)
    rank_loss = torch.mean(F.relu(pred_diff * gt_diff * pair_mask))
    loss = reg_loss + float(alpha) * rank_loss
    return loss, reg_loss, rank_loss


def price_return_rank_loss(price_prediction, base_price, ground_truth, mask, stock_num, alpha=0.1):
    price_prediction = _ensure_batched(price_prediction)
    base_price = _ensure_batched(base_price)
    denominator = torch.where(
        torch.abs(base_price) < 1e-12,
        torch.full_like(base_price, 1e-12),
        base_price,
    )
    return_ratio = (price_prediction - base_price) / denominator
    loss, reg_loss, rank_loss = return_rank_loss(return_ratio, ground_truth, mask, stock_num, alpha=alpha)
    return loss, reg_loss, rank_loss, return_ratio
