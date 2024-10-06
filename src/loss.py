import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.gamma import Gamma


class GammaLoss(nn.Module):
    """
    Gamma distribution with shape alpha and rate beta
    """
    def __init__(self, k, theta):
        super(GammaLoss, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = torch.tensor([k]).to(device)
        self.beta = torch.tensor([1/theta]).to(device)

    def forward(self, outputs):
        # log likelihood of gamma distribution with shape alpha and rate beta
        loss = torch.mean( (self.alpha - 1) * torch.log(outputs) - self.beta * outputs +self.alpha * torch.log(self.beta) - torch.lgamma(self.alpha) )
        return -1 * loss


class _ECELoss(nn.Module):
    """
    Reference: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        mce = []
        avg_confidence_in_bin_list = []
        accuracy_in_bin_list = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce.append(torch.abs(avg_confidence_in_bin - accuracy_in_bin))

                avg_confidence_in_bin_list.append(avg_confidence_in_bin.item())
                accuracy_in_bin_list.append(accuracy_in_bin.item())
        mce.sort()
        return ece.item(), mce[-1].item()

class Top_ECELoss(nn.Module):
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(Top_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        predicted_labels = torch.unique(predictions).tolist()
        num_data = len(predictions)
        index_per_class = {label: (predictions == label).nonzero(as_tuple=True)[0] for label in predicted_labels}

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            for label in predicted_labels:
                index = index_per_class[label]
                in_bin = confidences[index].gt(bin_lower.item()) * confidences[index].le(bin_upper.item())
                prop_in_bin = in_bin.float().sum()/num_data
                # Calculated |confidence - accuracy| in each bin
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = accuracies[index][in_bin].float().mean()
                    avg_confidence_in_bin = confidences[index][in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece.item()

def BrierLoss(logits, labels):
    """
    Calculate the Brier score for multi-class classification.
    """
    n_classes = logits.size(1)
    true_one_hot = F.one_hot(labels, num_classes=n_classes).to(torch.float32)
    probs = F.softmax(logits, dim=1)
    squared_diff = (probs - true_one_hot) ** 2
    score = squared_diff.mean()
    return score.item()