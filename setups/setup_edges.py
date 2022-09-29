import numpy as np

from base.datasets.dataset_base import EdgeCriterionBuilder


class CovidICUEdgeCriterion(EdgeCriterionBuilder):
    def __init__(self):
        super(CovidICUEdgeCriterion, self).__init__()

    @staticmethod
    def edge_criterion(meta_data, meta_col):
        threshold = [10, 0, 0]  # age, severity1, severity2
        dist = np.abs(meta_data[:, meta_col] - meta_data[:, meta_col, None])
        edges = dist <= threshold[meta_col]
        return edges.numpy()


class CovidDeathEdgeCriterion(EdgeCriterionBuilder):
    def __init__(self):
        super(CovidDeathEdgeCriterion, self).__init__()

    @staticmethod
    def edge_criterion(meta_data, meta_col):
        threshold = [10, 0, 0, 0.5]  # age, severity1, severity2, severity_mean
        dist = np.abs(meta_data[:, meta_col] - meta_data[:, meta_col, None])
        edges = dist <= threshold[meta_col]
        return edges.numpy()
