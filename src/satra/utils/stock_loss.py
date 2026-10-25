"""Supervised return-regression and cross-sectional ranking objective."""

import torch
import torch.nn.functional as F


def _ensure_batched(tensor):
    if tensor.dim() == 3:
        return tensor
    raise ValueError(f"expected [batch,stocks,1], got {tuple(tensor.shape)}")


def stockmixer_loss(
    prediction,
    ground_truth,
    base_price,
    mask,
    stock_num,
    alpha,
    prediction_target="price",
):
    """Combine masked return MSE with a pairwise stock-ranking penalty.

    All inputs use ``[B, N, 1]``. ``prediction`` is interpreted directly as a
    return when ``prediction_target='return'``; price predictions are converted
    relative to ``base_price`` first. The ranking term penalizes stock pairs
    whose predicted and observed return orderings disagree.
    """

    prediction = _ensure_batched(prediction)
    ground_truth = _ensure_batched(ground_truth)
    base_price = _ensure_batched(base_price)
    mask = _ensure_batched(mask)

    if (
        prediction.shape != ground_truth.shape
        or prediction.shape != base_price.shape
        or prediction.shape != mask.shape
    ):
        raise ValueError(
            "prediction, ground_truth, base_price and mask must share shape, got "
            f"{tuple(prediction.shape)}, {tuple(ground_truth.shape)}, {tuple(base_price.shape)}, {tuple(mask.shape)}"
        )
    if prediction.shape[1] != int(stock_num):
        raise ValueError(
            f"stock dimension mismatch: got {prediction.shape[1]}, expected {stock_num}"
        )

    if prediction_target == "return":
        return_ratio = prediction
    elif prediction_target == "price":
        return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    else:
        raise ValueError(f"Unsupported prediction_target: {prediction_target}")
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)

    rr = return_ratio.squeeze(-1)
    gt = ground_truth.squeeze(-1)
    valid = mask.squeeze(-1)
    pre_pw_dif = rr.unsqueeze(2) - rr.unsqueeze(1)
    gt_pw_dif = gt.unsqueeze(1) - gt.unsqueeze(2)
    mask_pw = valid.unsqueeze(2) * valid.unsqueeze(1)
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))

    loss = reg_loss + alpha * rank_loss

    return loss, reg_loss, rank_loss, return_ratio
