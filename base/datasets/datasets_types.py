import torch

from base.datasets.dataset_base import DatasetBuilder


class BatchDataset(DatasetBuilder):

    def __init__(self, config):
        super(BatchDataset, self).__init__(config)

    def generate_adj(self, batch_idx, fold, meta_col=0):
        all_adjs = torch.clone(self.init_all_adjs)
        # Take nodes of interest and remove the connections between validation
        # and test samples
        adj = all_adjs[meta_col, batch_idx.long(), :]
        adj = adj[:, batch_idx.long()]
        return adj


class FullDatasetLearn(DatasetBuilder):

    def __init__(self, config):
        super(FullDatasetLearn, self).__init__(config)

    def _set_adj(self, new_adj):
        pass

    def __getitem__(self, index):
        pass


class BatchDatasetLearn(DatasetBuilder):

    def __init__(self, config):
        super(BatchDatasetLearn, self).__init__(config)

    def _sample_adj(self, adj, index):
        pass

    def _set_adj(self, new_adj, idx):
        pass

    def __getitem__(self, index):
        pass


def get_dataset_type(datatype):
    if datatype == 'full':
        return BatchDataset
    elif datatype == 'batch':
        return BatchDataset
    elif datatype == 'full_learn':
        return FullDatasetLearn
    elif datatype == 'batch_learn':
        return BatchDatasetLearn
    else:
        raise ValueError('Dataset type {} is unknown.'.format(datatype))
