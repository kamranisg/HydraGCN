import torch

from base.models.models import ModelBuilder


class NanToZero(ModelBuilder):
    """Impute nan entries with zero-value and return indices of missing entries.

    When information pass through this model, an additional nan_idx keyname is added to the output dict to
    determine which values were initially imputed.

    # TODO
    """
    def __init__(self, conf, idx):
        super(NanToZero, self).__init__()
        assert conf.layers == 1, 'PyImpute model only needs one layer.'
        for layer in range(conf.layers):
            # self.kwargs = self.set_kwargs(conf, layer, idx)
            self.kwargs = conf['layer{}'.format(layer)]
            if 'dropout' in self.kwargs:
                self.dropout = torch.nn.Dropout(p=self.kwargs.get('dropout', .5))

    def forward(self, x_dict):
        in_x, adj = x_dict['input'], x_dict['adj']
        nan_idx = torch.isnan(in_x)
        out = torch.where(~nan_idx, in_x,  torch.zeros_like(in_x))
        if 'dropout' in self.kwargs:
            out = self.dropout(out)
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = out
        out_dict['nan_idx'] = nan_idx
        return out_dict


class MeanImpute(NanToZero):
    def __init__(self, conf, idx):
        super(MeanImpute, self).__init__(conf, idx)

    def forward(self, x_dict):
        x_dict = super(MeanImpute, self).forward(x_dict)
        x = x_dict['input']
        x_mean = torch.mean(x, dim=list(self.kwargs.get('dim')), keepdim=True)
        out = torch.where(~x_dict['nan_idx'], x, x_mean)
        if 'dropout' in self.kwargs:
            out = self.dropout(out)
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = out
        return out_dict
