import torch


def autocorr_coefficient(x: torch.Tensor, lag: int = 1) -> torch.Tensor:
    """
    Compute the autocorrelation coefficient of each sequence in batch at a given lag.

    Args:
        x: Tensor of shape (batch_size, seq_len), binary or real-valued spike train.
        lag: time lag for autocorrelation.

    Returns:
        Tensor of shape (batch_size,) with autocorrelation coefficients.
    """
    # mean over time
    mu = x.mean(dim=1, keepdim=True)
    x_centered = x - mu

    # numerator: sum_{t=0 to T-lag-1} (x_t - mu)(x_{t+lag}-mu)
    num = (x_centered[:, :-lag] * x_centered[:, lag:]).sum(dim=1)
    # denominator: sum_{t=0 to T-1} (x_t - mu)^2
    den = (x_centered ** 2).sum(dim=1)

    # avoid division by zero
    eps = 1e-8
    return num / (den + eps)