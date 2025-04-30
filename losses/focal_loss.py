import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceFocalLoss(nn.Module):
    def __init__(self, smooth=1e-6, alpha=0.8, gamma=2.0, dice_weight=0.6):
        """
        Combined Dice and Focal Loss
        
        Args:
            smooth: Smoothing factor for Dice
            alpha: Weighting factor for Focal Loss (class balance)
            gamma: Focusing parameter for Focal Loss
            dice_weight: Ratio of Dice to Focal Loss (0.6 means 60% Dice + 40% Focal)
        """
        super(DiceFocalLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        # Dice Loss Calculation
        pred_dice = pred.contiguous().view(-1)
        target_dice = target.contiguous().view(-1)
        
        intersection = (pred_dice * target_dice).sum()
        dice = (2. * intersection + self.smooth) / (pred_dice.sum() + target_dice.sum() + self.smooth)
        dice_loss = 1 - dice

        # Focal Loss Calculation
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t)**self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        # Combined Loss
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * focal_loss