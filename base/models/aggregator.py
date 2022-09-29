from torch import nn
import torch

from base.utils.configuration import Config


class AggregatorBuilder(nn.Module):
    def __init__(self, conf):
        super(AggregatorBuilder, self).__init__()
        self.conf = conf

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = str(cls.__name__)
        Config.global_dict['model'][name] = cls

    def forward(self, x):
        raise NotImplementedError


class Concat(AggregatorBuilder):
    def __init__(self, conf):
        super(Concat, self).__init__(conf)

    def forward(self, x):
        input_list = []
        output = dict()
        output['adj'] = x[-1]['adj']  # Just take a single adjacency matrix
        output['adj_wts'] = x[-1].get('adj_wts', None)  # Just take a single weighted adjacency matrix
        output['nan_idx'] = x[-1].get('nan_idx', None)  # Just take a single masking matrix
        for inp in x:
            input_list.append(inp['input'])
        # TODO handle inputs with tensor shape > 2
        if 'dim' in self.conf:
            agg_dim = self.conf.dim
        else:
            agg_dim = -1
        output['input'] = torch.cat(input_list, dim=agg_dim)
        # Pass on additional items not covered yet:
        for x_dict in x:
            for k, v in x_dict.items():
                if k not in ['input', 'adj', 'adj_wts', 'nan_idx']:
                    output[k] = v
        return output


class Average(AggregatorBuilder):
    def __init__(self, conf):
        super(Average, self).__init__(conf)

    def forward(self, x):
        input_list = []
        output = dict()
        output['adj'] = x[-1]['adj']  # Just take a single adjacency matrix
        output['adj_wts'] = x[-1].get('adj_wts', None)  # Just take a single weighted adjacency matrix
        output['nan_idx'] = x[-1].get('nan_idx', None)  # Just take a single masking matrix
        dim = x[0]['input'].shape[-1]
        for inp in x:
            if not inp['input'].shape[-1] == dim:
                raise ValueError('For Average Aggregator, all input dimensions must match in last dimension')
            input_list.append(inp['input'])

        # TODO handle inputs with tensor shape > 2
        if 'dim' in self.conf:
            agg_dim = self.conf.dim
        else:
            agg_dim = -1
        if len(input_list) == 1:
            output['input'] = torch.mean(input_list[0], dim=agg_dim)
        else:
            input_tensor = torch.stack(input_list, agg_dim)
            output['input'] = torch.mean(input_tensor, dim=agg_dim)
        return output


class Sum(AggregatorBuilder):
    def __init__(self, conf):
        super(Sum, self).__init__(conf)
        self.conf = conf

    def forward(self, x):
        input_list = []
        output = dict()
        output['adj'] = x[-1]['adj']  # Just take a single adjacency matrix
        output['adj_wts'] = x[-1].get('adj_wts', None)  # Just take a single weighted adjacency matrix
        output['nan_idx'] = x[-1].get('nan_idx', None)  # Just take a single masking matrix
        dim = x[0]['input'].shape[-1]
        for inp in x:
            if not inp['input'].shape[-1] == dim:
                raise ValueError('For Sum Aggregator, all input dimensions must match in last dimension')
            input_list.append(inp['input'])
        # TODO handle inputs with tensor shape > 2
        if 'dim' in self.conf:
            agg_dim = self.conf.dim
        else:
            agg_dim = -1
        if len(input_list) == 1:
            output['input'] = torch.mean(input_list[0], dim=agg_dim)
        else:
            input_tensor = torch.stack(input_list, agg_dim)
            output['input'] = torch.mean(input_tensor, dim=agg_dim)
        return output


class Difference(AggregatorBuilder):
    def __init__(self, conf):
        super(Difference, self).__init__(conf)

    def forward(self, x):
        if not len(x) == 2:
            raise ValueError('Difference Aggregator needs exactly 2 inputs, but received {}'.format(len(x)))
        input_list = []
        output = dict()
        output['adj'] = x[-1]['adj']  # Just take a single adjacency matrix
        output['adj_wts'] = x[-1].get('adj_wts', None)  # Just take a single weighted adjacency matrix
        output['nan_idx'] = x[-1].get('nan_idx', None)  # Just take a single masking matrix
        dim = x[0]['input'].shape[-1]
        for inp in x:
            if not inp['input'].shape[-1] == dim:
                raise ValueError('For Difference Aggregator, the two input dimensions must match in last dimension')
            input_list.append(inp['input'])
        # TODO handle inputs with tensor shape > 2
        output['input'] = input_list[0] - input_list[1]
        return output


class Pool(AggregatorBuilder):
    def __init__(self, conf):
        super(Pool, self).__init__(conf)

    def forward(self, x):
        input_list = []
        output = dict()
        output['adj'] = x[-1]['adj']  # Just take a single adjacency matrix
        output['adj_wts'] = x[-1].get('adj_wts', None)  # Just take a single weighted adjacency matrix
        output['nan_idx'] = x[-1].get('nan_idx', None)  # Just take a single masking matrix
        dim = x[0]['input'].shape[-1]
        for inp in x:
            if not inp['input'].shape[-1] == dim:
                raise ValueError('For Pool Aggregator, all input dimensions must match in last dimension')
            input_list.append(inp['input'])
        # TODO handle inputs with tensor shape > 2
        if 'dim' in self.conf:
            agg_dim = self.conf.dim
        else:
            agg_dim = -1
        input_tensor = torch.stack(input_list, agg_dim)
        output['input'], _ = torch.max(input_tensor, dim=agg_dim)
        return output


class Pass(AggregatorBuilder):
    def __init__(self, conf):
        super(Pass, self).__init__(conf)

    def forward(self, x):
        x_dict = x[0]
        # x -> [{'keyname': torch_tensor}]
        output = {k: v for k, v in x_dict.items()}
        return output


class Stack(AggregatorBuilder):
    def __init__(self, conf):
        super(Stack, self).__init__(conf)

    def forward(self, x):
        input_list = []
        output = dict()
        output['adj'] = x[-1]['adj']  # Just take a single adjacency matrix
        output['adj_wts'] = x[-1].get('adj_wts', None)  # Just take a single weighted adjacency matrix
        output['nan_idx'] = x[-1].get('nan_idx', None)  # Just take a single masking matrix
        for inp in x:
            input_list.append(inp['input'])
        # TODO handle inputs with tensor shape > 2
        if 'dim' in self.conf:
            agg_dim = self.conf.dim
        else:
            agg_dim = 1
        output['input'] = torch.stack(input_list, dim=agg_dim)
        return output


class Multiply(AggregatorBuilder):
    """#TODO Element-wise multiplication of two tensors with the same tensor length"""
    def __init__(self, conf):
        super(Multiply, self).__init__(conf)

    def forward(self, x):
        if not len(x) == 2:
            raise ValueError('Multiply Aggregator needs exactly 2 inputs, but received {}'.format(len(x)))
        input_list = []
        output = dict()
        output['adj'] = x[-1]['adj']  # Just take a single adjacency matrix
        output['adj_wts'] = x[-1].get('adj_wts', None)  # Just take a single weighted adjacency matrix
        output['nan_idx'] = x[-1].get('nan_idx', None)  # Just take a single masking matrix
        for inp in x:
            input_list.append(inp['input'])
        # TODO handle inputs with tensor shape > 2
        output['input'] = input_list[0] * input_list[1]
        return output


class Merge(AggregatorBuilder):
    """Given two dictionaries merge all elements into one dictionary."""
    def __init__(self, conf):
        super(Merge, self).__init__(conf)
        # self.keyname0 = conf.get('keyname0', 'pred_tensor')
        self.keyname1 = conf.get('keyname1', 'target_tensor')

    def forward(self, x):
        assert len(x) == 2, "List of input dict should be two instead it contains {} element".format(len(x))
        # output = dict()
        pred = x[0]
        target = x[1]
        output = {k: v for k, v in pred.items()}
        # output[self.keyname0] = pred['input']
        output[self.keyname1] = target['input']
        return output


class Clone(AggregatorBuilder):
    """Clone using given input keyname with new keyname1."""
    def __init__(self, conf):
        super(Clone, self).__init__(conf)
        self.keyname0 = conf.get('keyname0', 'input')
        self.keyname1 = conf.get('keyname1')

    def forward(self, x):
        input_dict = x[0]
        input_dict[self.keyname1] = input_dict[self.keyname0].clone()
        return input_dict
