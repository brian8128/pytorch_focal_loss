
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean', gamma=0):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.gamma = gamma
        
    def forward(self, input_, target):
        ignored = target == self.ignore_index              # mb, d1, d2, ..., dk
        # Set the ignored labels to zero. We will later multiply these by zero
        # weights to ignore them. 
        target = target.clone()
        target[ignored] = 0
        ignored = ignored.type(torch.FloatTensor)
        logp = F.log_softmax(input_, dim=1)                # mb, C, d1, d2, ..., dk
        # Gather the predictions for the true labels
        logpt = torch.gather(logp, 1, target.unsqueeze(1)) # mb, 1, d1, d2, ..., dk
        logpt = logpt.squeeze(1)                           # mb, d1, d2, ..., dk
        if self.weight is not None:
            w = self.weight.expand(target.shape + self.weight.shape) # mb, d1, d2, ..., dk, C
            # Construct the permutation that will move the channels from the end to
            # index 1. There has got to be an easier way
            permutation = (0, -1) + tuple(range(1, len(w.shape)-1)) 
            w = w.permute(permutation)                               # mb, C, d1, d2, ..., dk
            # Gather the weights for the true labels.
            wt = torch.gather(w, 1, target.unsqueeze(1))             # mb, 1 d1, d2, ..., dk
            wt = wt.squeeze(1)                                       # mb, d1, d2, ..., dk
            wt *= 1 - ignored                                        # mb, d1, d2, ..., dk
        else:
            wt = 1 - ignored
        loss = - wt * (1-torch.exp(logpt)) ** self.gamma * logpt
        if self.reduction == 'mean':
            return torch.sum(loss / torch.sum(wt))
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
