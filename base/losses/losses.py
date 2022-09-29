import torch
import torch_geometric as tg
from torch import nn
from base.utils.configuration import Config
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing


class ClassificationLossBuilder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ClassificationLossBuilder, self).__init__()
        self.mode = 'Classification'
        self.pred_keyname = kwargs.get('keyname', 'input')
        self.target_keyname = kwargs.get('target_keyname', None)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = str(cls.__name__)
        Config.global_dict['losses'][name] = cls

    def forward(self, pred, target):
        indices = pred['indices']

        if self.target_keyname is not None:
            # If loss is calculated from initial/intermediate tensors within the pipeline.
            cur_loss = self.loss(pred[self.pred_keyname][indices], pred[self.target_keyname][indices])
        else:
            print("-----------------losses.py-----------------------")
            print(pred[self.pred_keyname][indices])
            print(pred[self.pred_keyname][indices].shape)
            print("TARGETS")
            print(target[indices])
            print(target[indices].shape)
            
            # Reshaping added for ham10k by Kamran
            a=target[indices]
            b=a.reshape(a.shape[0])

            cur_loss = self.loss(pred[self.pred_keyname][indices], b)

        return cur_loss


class RegressionLossBuilder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RegressionLossBuilder, self).__init__()
        self.mode = 'Regression'
        self.pred_keyname = kwargs.get('keyname', 'input')
        self.target_keyname = kwargs.get('target_keyname', None)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = str(cls.__name__)
        Config.global_dict['losses'][name] = cls

    def forward(self, pred, target):
        indices = pred['indices']

        if self.target_keyname is not None:
            # If loss is calculated from initial/intermediate tensors within the pipeline.
            cur_loss = self.loss(pred[self.pred_keyname][indices], pred[self.target_keyname][indices])

            # print('pred and target min', pred[self.pred_keyname][indices].min(), pred[self.target_keyname][
            #     indices].min())
            # print('pred and target max', pred[self.pred_keyname][indices].max(), pred[self.target_keyname][
            #     indices].max())

        else:
            cur_loss = self.loss(pred[self.pred_keyname][indices], target[indices])

        return cur_loss


class CrossEntropyLoss(ClassificationLossBuilder):
    def __init__(self, loss_dict):
        super(CrossEntropyLoss, self).__init__(**loss_dict)
        print("CCCCCCRRRRRREEEEEEEENNNNNLOOOOOOOOSS")
        if 'kwargs' in loss_dict.keys():
            kwargs = loss_dict['kwargs']
        else:
            # TODO weight balancing
            kwargs = {'reduction': 'mean'}
            # kwargs = {'weights': mfb_weights,
            #           'size_average': None}
        self.loss = nn.CrossEntropyLoss(**kwargs)

    # def forward(self, pred, target):
    #     return self.loss(pred['input'], target)


class MSELoss(RegressionLossBuilder):
    def __init__(self, loss_dict):
        super(MSELoss, self).__init__(**loss_dict)
        if 'kwargs' in loss_dict.keys():
            kwargs = loss_dict['kwargs']
        else:
            # TODO
            kwargs = {'reduction': 'mean'}
        self.loss = nn.MSELoss(**kwargs)

    # def forward(self, pred, target):
    #     return self.loss(pred['input'], target)


class L1Loss(RegressionLossBuilder):
    def __init__(self, loss_dict):
        super(L1Loss, self).__init__(**loss_dict)
        if 'kwargs' in loss_dict.keys():
            kwargs = loss_dict['kwargs']
        else:
            # TODO
            kwargs = {'reduction': 'mean'}
        self.loss = nn.L1Loss(**kwargs)

    # def forward(self, pred, target):
    #     return self.loss(pred['input'], target)


class BCELoss(RegressionLossBuilder):
    def __init__(self, loss_dict):
        super(BCELoss, self).__init__(**loss_dict)
        if 'kwargs' in loss_dict.keys():
            kwargs = loss_dict['kwargs']
        else:
            # TODO
            kwargs = {'reduction': 'mean'}
        self.loss = nn.BCELoss(**kwargs)

    # def forward(self, pred, target):
    #     return self.loss(pred['input'], target)


class FrobeniusNorm(RegressionLossBuilder):
    """#TODO Squared Frobenius norm"""
    def __init__(self, loss_dict):
        super(FrobeniusNorm, self).__init__(**loss_dict)
        # self.loss = torch.norm

    def forward(self, pred, target):
        #print("FROB Norm--------------------")

        indices = pred['indices']  # TODO Check if this works!!!

        nan_idx = ~pred['nan_idx'][indices]

        mask = torch.zeros(nan_idx.shape)

        mask[0:10] = 1

        #print("pr tr")
        pr = pred['input'][indices][:, 0:10]
        tr = target[indices][:, 0:10]
        #print(pr.shape, tr.shape)

        m1 = torch.mean(tr.float(), 0)

        am1 = (tr - m1).float()

        var1 = torch.std(tr.float(), 0)

        std_mean1 = (am1 / var1).float()
        df = ((pr - std_mean1) ** 2).sum()

        m = torch.mean(target[indices].float(), 0)

        am = (target[indices] - m).float()

        var = torch.std(target[indices].float(), 0)

        std_mean = (am / var).float()

        diff_mat = (pred['input'][indices] * nan_idx - target[indices] * nan_idx) ** 2

        d2 = (pred['input'][indices] * nan_idx - std_mean * nan_idx) ** 2
        print("Frob norm"+str(df))
        return df

    #def forward(self, pred, target):
    #    # Recheck calculation
    #    indices = pred['indices']  # TODO Check if this works!!!
    #    nan_idx = ~pred['nan_idx'][indices]
    #    diff_mat = (pred['input'][indices] * nan_idx - target[indices] * nan_idx)**2
    #    return diff_mat.sum()


class DirichletNorm(RegressionLossBuilder):
    def __init__(self, loss_dict):
        super(DirichletNorm, self).__init__(**loss_dict)


    def forward(self, pred, target):
        
        return 0.01
        #print("DIR NORM")
        indices = pred['indices']  # TODO Check if this works!!!
        X = pred['input'][indices]

        y = tg.utils.convert.to_scipy_sparse_matrix(pred['adj'])

        y1 = y.todense()
        y1 = y1[:X.shape[0], :X.shape[0]]
        y = sp.coo_matrix(y1)
        row_sum = y.sum(axis=1).reshape(1, y.shape[0])
        offsets = np.array([0])
        D = sp.dia_matrix((row_sum, offsets), shape=(y.shape[0], y.shape[0]))

        L = D - y

        #L = sp.identity(D.shape[0]) - (D.power(-1 / 2)) * y * (D.power(-1 / 2))
        # print(norm_laplacian)

        laplacian_dense = L.todense()

        u = preprocessing.normalize(laplacian_dense, norm='l2')
        
        
        q = torch.from_numpy(u).to('cuda').float() # enter device name here
        #q = torch.from_numpy(u).float()
        l1 = torch.matmul(X.T, q)
        l2 = torch.matmul(l1, X)
        #print(torch.trace(l2))

        return torch.trace(l2)

        #indices = pred['indices']  # TODO Check if this works!!!
        #X = pred['input'][indices]
        #L = tg.utils.get_laplacian(pred['adj'][indices])
        #norm_laplacian = tg.utils.to_dense_adj(L[0], batch=None, edge_attr=L[1])[0]
        #res_matrix = torch.matmul(torch.matmul(torch.transpose(X, 0, 1), norm_laplacian), X)
        #return torch.trace(res_matrix)


class L1Norm(RegressionLossBuilder):
    def __init__(self, loss_dict):
        super(L1Norm, self).__init__(**loss_dict)
        if 'kwargs' in loss_dict.keys():
            self.kwargs = loss_dict['kwargs']
        else:
            # TODO
            self.kwargs = {'p': 1, 'dim': 1}
        self.loss = torch.norm

    def forward(self, pred, target):
        indices = pred['indices']
        cur_loss = self.loss(pred[self.pred_keyname][indices], **self.kwargs).mean()
        return cur_loss
