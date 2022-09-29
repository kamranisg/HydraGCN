import torch
from torch import nn
from torch.nn import functional as F

from base.utils.configuration import Config


class CalculatePredictionBuilder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CalculatePredictionBuilder, self).__init__()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = str(cls.__name__)
        Config.global_dict['train'][name] = cls

    def forward(self, *args):
        raise NotImplementedError


class ClassificationPred(CalculatePredictionBuilder):
    def __init__(self, *args, **kwargs):
        super(CalculatePredictionBuilder, self).__init__()

    def forward(self, preds):
        preds_dict = {}
        # if len(preds) > 1:
        #     raise NotImplementedError  # TODO has to be adapted to multiple outputs
        # else:
        preds = torch.tensor(preds)
        if preds.shape[-1] > 1:
            # Collect output labels and probabilities for saving.
            preds_dict['proba'] = torch.nn.functional.softmax(preds, dim=-1)
            preds_dict['pred'] = torch.argmax(preds_dict['proba'], -1)
        else:
            preds_dict['proba'] = F.sigmoid(preds)
            preds_dict['pred'] = torch.round(preds_dict['proba'])
        return preds_dict


class RegressionPred(CalculatePredictionBuilder):
    def __init__(self, *args, **kwargs):
        super(RegressionPred, self).__init__()

    def forward(self, preds):
        preds_dict = {'pred': torch.tensor(preds)}
        return preds_dict
