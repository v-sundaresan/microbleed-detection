from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

#==============================#
# Microbleednet loss functions #
# Vaanathi Sundaresan          #
# 09-01-2023                   #
#==============================#

def calculate_dice_coefficient(prediction, target):

    smooth = 1.0
    prediction_vector = prediction.view(-1)
    target_vector = target.view(-1)

    intersection = intersection = (prediction_vector * target_vector).sum()
    dice = (2.0 * intersection + smooth) / (prediction_vector.sum() + target_vector.sum() + smooth)
    return dice

class DiceLoss(nn.Module):
    
    def __init__(self, weight=None):
        super().__init__()

    def calculate_dice_coefficient(self, prediction, target):

        smooth = 1.0
        prediction_vector = prediction.reshape(-1)
        target_vector = target.reshape(-1)

        intersection = (prediction_vector * target_vector).sum()
        dice = (2.0 * intersection + smooth) / (prediction_vector.sum() + target_vector.sum() + smooth)
        return dice

    def forward(self, binary_prediction, binary_target):
        """
        Forward pass
        :param binary_prediction: torch.tensor (NxCxHxW)
        :param binary_target: torch.tensor (NxHxW)
        :return: scalar
        """
        
        return 1 - self.calculate_dice_coefficient(binary_prediction, binary_target)


class CrossEntropyLoss(nn.Module):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight, reduction='mean')

    def forward(self, inputs, targets):
        """
        Forward pass
        :param inputs: torch.tensor (NxC)
        :param targets: torch.tensor (N)
        :return: scalar
        """
        
        targets = targets.to(inputs.device)
        return self.loss(inputs, targets)


class CombinedLoss(nn.Module):
    """
    A combination of dice and cross entropy loss
    """

    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.cross_entropy_loss = CrossEntropyLoss()

    def forward(self, input, target, weight=None):
        """
        Forward pass
        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW), optional
        :return: scalar
        """
        
        probabilities_vector = F.softmax(input, dim=1)
        # mask_vector = (probabilities_vector > 0.5).double()

        dice_loss = self.dice_loss(probabilities_vector[:, 1], target[:, 1])
        ce_loss = self.cross_entropy_loss(input, target)

        if weight is not None:
            ce_loss = ce_loss * weight.view(-1).cuda()

        dice_loss = dice_loss.mean()
        ce_loss = ce_loss.mean()

        return dice_loss + ce_loss
        # return ce_loss


class DistillationLoss(nn.Module):
    """
    A combination of dice and cross entropy loss for knowledge distillation
    """

    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.cross_entropy_loss = CrossEntropyLoss()

    def forward(self, input, teacher_scores, target, temperature, alpha):
        """
        Forward pass
        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param temperature: scalar
        :param alpha: scalar
        :param teacher_scores: torch.array
        :return: scalar
        """

        p = F.log_softmax(input / temperature, dim=1)
        q = F.log_softmax(teacher_scores / temperature, dim=1)

        ce_loss = F.cross_entropy(input, target)
        kl_divergence_loss = F.kl_div(p, q, log_target=True, reduction='batchmean') * (temperature ** 2)

        return (kl_divergence_loss * alpha + ce_loss * (1.0 - alpha))



