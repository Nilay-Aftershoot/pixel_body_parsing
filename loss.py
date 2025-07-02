#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import sys


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))#.cuda
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

# resulted in nans
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, eps=1e-6, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps   = eps
        self.nll   = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
         # ----------- debug hook: detect negative targets -------------------
        """lbl_cpu = labels.detach().cpu()
        lgt_cpu = logits.detach().cpu()
        print(np.unique(lbl_cpu))
        print(np.unique(lgt_cpu))
        print(lgt_cpu.shape)
        if (lbl_cpu < 0).any():
            print("Hahahah")
            sys.exit(0)
            bad_mask  = labels < 0
            uniq_vals = torch.unique(labels[bad_mask]).tolist()
            # Show up to first 10 pixel coordinates so you can trace them
            sample_px = torch.nonzero(bad_mask, as_tuple=False)[:10].tolist()

            print(
                f"[SoftmaxFocalLoss-DEBUG] "
                f"{bad_mask.sum().item()} pixels carry NEGATIVE label ids "
                f"(unique values: {uniq_vals}). "
                f"Sample locations (tensor indices): {sample_px}"
            )
            # <no fixes here – we want the underlying kernel to explode>"""
        # ------------------------------------------------------------------
        scores = F.softmax(logits, dim=1)
        # factor = torch.pow(1.-scores, self.gamma)
        factor    = torch.pow(torch.clamp_min(1.0 - scores, self.eps), self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        logits = torch.flatten(logits, 1)
        labels = torch.flatten(labels, 1)

        intersection = torch.sum(logits * labels, dim=1)
        loss = 1 - ((2 * intersection + self.smooth) / (logits.sum(1) + labels.sum(1) + self.smooth))

        return torch.mean(loss)



class DetailLoss(nn.Module):
    '''Implement detail loss used in paper
       `Rethinking BiSeNet For Real-time Semantic Segmentation`'''
    def __init__(self, dice_loss_coef=1., bce_loss_coef=1., smooth=1):
        super().__init__()
        self.dice_loss_coef = dice_loss_coef
        self.bce_loss_coef = bce_loss_coef
        self.dice_loss_fn = DiceLoss(smooth)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        loss = self.dice_loss_coef * self.dice_loss_fn(logits, labels) + \
               self.bce_loss_coef * self.bce_loss_fn(logits, labels)

        return loss



if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()
