import torch

def masked_mae_loss(y_pred, y_true):
    y_true[y_true < 1e-4] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean() # assign the sample weights of zeros to nonzero-values 
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mape_loss(y_pred, y_true):
    y_true[y_true < 1e-4] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean() 
    loss = torch.abs((y_pred - y_true) / y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()

def masked_rmse_loss(y_pred, y_true):
    y_true[y_true < 1e-4] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean() 
    loss = torch.pow(y_pred - y_true, 2)
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())
