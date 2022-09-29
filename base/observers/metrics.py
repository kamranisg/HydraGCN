import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.nn import functional as F

from base.utils.configuration import Config


class MetricsBuilder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MetricsBuilder, self).__init__()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = str(cls.__name__)
        Config.global_dict['train'][name] = cls

    def forward(self, *args):
        raise NotImplementedError


class ClassificationMetrics(MetricsBuilder):
    def __init__(self, *args, **kwargs):
        super(ClassificationMetrics, self).__init__()

    def forward(self, epoch_loss, outputs, labels, phase, config, idx):
        # if len(outputs) > 1:
        #     raise NotImplementedError  # TODO has to be adapted to multiple outputs
        # else:
        outputs = torch.tensor(outputs)
        if len(list(outputs.shape)) == 1:
            outputs = F.sigmoid(outputs)
            preds = torch.round(outputs)
        else:
            _, preds = torch.max(F.log_softmax(outputs, dim=1), 1)
        labels = torch.tensor(labels)
        epoch_acc = accuracy_score(labels, preds)
        epoch_f1 = f1_score(labels, preds, average='macro')

        # changed to work for binary classification
        confusion_matrix = torch.zeros((max(config.args.target_shape_list[idx],2),
                                        max(config.args.target_shape_list[idx], 2)), dtype=torch.int64)

        for t, p in zip(labels, preds):
            confusion_matrix[t.long(), p.long()] += 1

        print('{} Loss {}: {:.4f} Acc: {:.4f} F1 Score: {:.4f}'.format(phase, idx, epoch_loss, epoch_acc, epoch_f1))
        print('Confusion matrix (row=target, column=preds):')
        print(confusion_matrix)

        confusion_matrix = confusion_matrix.type(torch.float32)
        print('Class accuracy:')
        class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        print(class_acc)
        print()

        class_acc = class_acc.numpy()
        conf_mat = confusion_matrix.numpy()

        epoch_metrics = {'loss': epoch_loss,
                         'acc': epoch_acc,
                         'f1': epoch_f1,
                         'class_acc': class_acc,
                         'conf_mat': conf_mat,
                         'checkpointer': epoch_f1}

        return epoch_metrics


class RegressionMetrics(MetricsBuilder):
    def __init__(self, *args, **kwargs):
        super(RegressionMetrics, self).__init__()

    def forward(self, epoch_loss, outputs, labels, phase, config, idx):
        outputs = torch.tensor(outputs)
        labels = torch.tensor(labels)
        mae = torch.sum(torch.abs(outputs - labels)) / (1.0 * labels.size()[0])
        rmse = np.sqrt(epoch_loss)
        print('{} Loss {}: {:.4f} RMSE: {:.4f} MAE: {:.4f}'.format(phase, idx, epoch_loss, rmse.item(), mae.item()))

        epoch_metrics = {'loss': epoch_loss,
                         'mae': mae.item(),
                         'rmse': rmse.item(),
                         'checkpointer': -epoch_loss}

        return epoch_metrics


class RegularizerMetric(MetricsBuilder):
    def __init__(self, *args, **kwargs):
        super(RegularizerMetric, self).__init__()

    def forward(self, epoch_loss, outputs, labels, phase, config, idx):
        # outputs = torch.tensor(outputs)
        # labels = torch.tensor(labels)
        # mae = torch.sum(torch.abs(outputs - labels)) / (1.0 * labels.size()[0])
        # rmse = np.sqrt(epoch_loss)
        # print('{} Loss {}: {:.4f} RMSE: {:.4f} MAE: {:.4f}'.format(phase, idx, epoch_loss, None,
        #                                                            None))

        epoch_metrics = {'loss': epoch_loss,
                         'checkpointer': -epoch_loss}

        return epoch_metrics


class MultiTaskClassifierMetric(MetricsBuilder):
    def __init__(self, *args, **kwargs):
        super(MultiTaskClassifierMetric, self).__init__()

    def forward(self, epoch_loss, outputs, labels, phase, config, idx):
        outputs = torch.tensor(outputs)
        if len(list(outputs.shape)) == 1:
            outputs = F.sigmoid(outputs)
            preds = torch.round(outputs)
        else:
            _, preds = torch.max(F.log_softmax(outputs, dim=1), 1)
        labels = torch.tensor(labels)
        epoch_acc = accuracy_score(labels, preds)
        epoch_f1 = f1_score(labels, preds, average='macro')

        confusion_matrix = torch.zeros((config.args.target_shape_list[idx],
                                        config.args.target_shape_list[idx]), dtype=torch.int64)

        for t, p in zip(labels, preds):
            confusion_matrix[t.long(), p.long()] += 1

        print('{} Loss {}: {:.4f} Acc: {:.4f} F1 Score: {:.4f}'.format(phase, idx, epoch_loss, epoch_acc, epoch_f1))
        print('Confusion matrix (row=target, column=preds):')
        print(confusion_matrix)

        confusion_matrix = confusion_matrix.type(torch.float32)
        print('Class accuracy:')
        class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        print(class_acc)
        print()

        class_acc = class_acc.numpy()
        conf_mat = confusion_matrix.numpy()

        epoch_metrics = {'loss': epoch_loss,
                         'acc': epoch_acc,
                         'f1': epoch_f1,
                         'class_acc': class_acc,
                         'conf_mat': conf_mat,
                         'checkpointer': epoch_loss}

        return epoch_metrics


# class SegmentationMetrics(MetricsBuilder):
#     # TODO
#     def __init__(self, *args, **kwargs):
#         raise NotImplementedError
