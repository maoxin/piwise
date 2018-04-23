import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        # self.loss = nn.NLLLoss2d(weight, ignore_index=0)
        self.weight = weight

    def forward(self, output, target, size_average=True):
        # print(F.log_softmax(outputs, dim=1).shape)
        # return self.loss(F.log_softmax(outputs), targets)

        n, c, h, w = output.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between output and target
        if h > ht and w > wt: # upsample labels
            target = target.unsequeeze(1)
            target = F.upsample(target, size=(h, w), mode='nearest')
            target = target.sequeeze(1)
        elif h < ht and w < wt: # upsample images
            output = F.upsample(output, size=(ht, wt), mode='bilinear')
        elif h != ht and w != wt:
            raise Exception("Only support upsampling")

        target = target - 1

        log_p = F.log_softmax(output, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target,
                        weight=self.weight, size_average=False)
        if size_average:
            if self.weight:
                loss /= self.weight[target.data].sum()
            else:
                loss /= mask.data.sum()

        return loss
