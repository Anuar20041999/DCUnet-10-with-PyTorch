import torch

def wSDRloss(noised, clean, pred):
    noise = noised - clean
    pred_noise = noised - pred

    abs_clean = clean.norm(dim=1)
    abs_pred = pred.norm(dim=1)
    abs_noise = noise.norm(dim=1)
    abs_pred_noise = pred_noise.norm(dim=1)

    alpha = abs_clean**2 / (abs_clean**2 + abs_noise**2)
    loss1 = (clean*pred).sum(dim=1) / (abs_clean*abs_pred)
    loss2 = (noise*pred_noise).sum(dim=1) / (abs_noise*abs_pred_noise)
    loss = -alpha * loss1 - (1-alpha) * loss2
    return loss.mean(), loss1.mean(), loss2.mean()
