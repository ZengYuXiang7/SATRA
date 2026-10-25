import torch
import torch.nn.functional as F

from .model import stock_rank_loss


def _ensure_batched(tensor):
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    if tensor.dim() == 4:
        return tensor
    raise ValueError(f"expected [stocks,seq,features] or [batch,stocks,seq,features], got {tuple(tensor.shape)}")


def _ensure_stock_value(tensor):
    if tensor.dim() == 1:
        return tensor.view(1, -1, 1)
    if tensor.dim() == 2:
        return tensor.unsqueeze(0) if tensor.size(-1) == 1 else tensor.unsqueeze(-1)
    if tensor.dim() == 3:
        return tensor
    raise ValueError(f"expected stock value tensor with dim 1-3, got {tuple(tensor.shape)}")


def masked_mse(prediction, target, mask):
    target = target.reshape_as(prediction)
    mask = mask.reshape_as(prediction).to(dtype=prediction.dtype)
    return ((prediction - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)


def enhancer_objective(
    model,
    inputs,
    target,
    mask,
    mse_weight=1.0,
    rank_weight=1.0,
    reg1_weight=12.0,
    reg2_weight=1.0,
    noise_scale=1e-3,
    prediction_key="prediction",
):
    inputs = _ensure_batched(inputs)
    target = _ensure_stock_value(target)
    mask = _ensure_stock_value(mask)
    if target.shape[0] != inputs.shape[0] or mask.shape[0] != inputs.shape[0]:
        raise ValueError(
            f"batch mismatch: inputs={tuple(inputs.shape)}, target={tuple(target.shape)}, mask={tuple(mask.shape)}"
        )

    losses = []
    mse_losses = []
    rank_losses = []
    reg1_losses = []
    reg2_losses = []
    predictions = []
    outputs = []
    for index in range(inputs.shape[0]):
        x_i = inputs[index]
        y_i = target[index].reshape(-1)
        mask_i = mask[index].reshape(-1)
        output = model(x_i)
        pred = output[prediction_key].reshape(-1)
        mse = masked_mse(pred, y_i, mask_i)
        rank = stock_rank_loss(pred, y_i, mask_i)
        reg1 = output["intervention_loss"].float()
        if noise_scale > 0.0 and reg2_weight > 0.0:
            noisy_x = x_i + torch.randn_like(x_i) * float(noise_scale)
            noisy_pred = model(noisy_x)[prediction_key].reshape(-1)
            reg2 = F.mse_loss(noisy_pred, pred.detach())
        else:
            reg2 = pred.new_zeros(())
        loss = (
            float(mse_weight) * mse
            + float(rank_weight) * rank
            + float(reg1_weight) * reg1
            + float(reg2_weight) * reg2
        )
        losses.append(loss.float())
        mse_losses.append(mse.float())
        rank_losses.append(rank.float())
        reg1_losses.append(reg1.float())
        reg2_losses.append(reg2.float())
        predictions.append(pred.view(-1, 1))
        outputs.append(output)

    return {
        "loss": torch.stack(losses).mean(),
        "mse_loss": torch.stack(mse_losses).mean(),
        "rank_loss": torch.stack(rank_losses).mean(),
        "reg1_intervention_loss": torch.stack(reg1_losses).mean(),
        "reg2_noisy_consistency_loss": torch.stack(reg2_losses).mean(),
        "prediction": torch.stack(predictions, dim=0),
        "outputs": outputs,
    }


@torch.no_grad()
def enhancer_predict(model, inputs, prediction_key="invariant_prediction"):
    inputs = _ensure_batched(inputs)
    predictions = []
    for index in range(inputs.shape[0]):
        output = model(inputs[index])
        predictions.append(output[prediction_key].reshape(-1, 1).float())
    return torch.stack(predictions, dim=0)
