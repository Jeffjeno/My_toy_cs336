import torch

def cross_entropy(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute mean cross-entropy for a batch of logits and integer targets.

    Args:
        x: Tensor of shape [batch_size, vocab_size], unnormalized logits.
        target: Tensor of shape [batch_size], integer class indices in [0, vocab_size-1].

    Returns:
        Scalar tensor: the average cross-entropy loss across the batch.

    Notes:
        Numerically stable via log-sum-exp trick: logsumexp(x) = m + log(sum(exp(x - m))),
        where m = max(x, dim=1, keepdim=True).
    """
    if target.dtype != torch.long:
        target = target.long()

    # maximum 
    m = x.max(dim=1, keepdim=True).values                 # [B, 1]
    x_shift = x - m                                       # [B, V]


    lse = m.squeeze(1) + torch.log(torch.exp(x_shift).sum(dim=1))  # [B]

    # Gather 
    correct = x.gather(1, target.view(-1, 1)).squeeze(1)  # [B]

    
    loss = lse - correct                                  # [B]
    return loss.mean()                                    # scalar