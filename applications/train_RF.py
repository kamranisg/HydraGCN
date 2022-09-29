from setups.setup_data import data_setup
from base.datasets.datasets_types import get_dataset_type
import argparse
import numpy as np
from base.datasets.dataset_base import CustomDataLoader
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='COVID19_ICU_Length', help='Specifies the used dataset.')
parser.add_argument('--datatype', type=str, default='full', help='Specifies the used type of dataset.')
parser.add_argument('--train_mode', type=str, default='regression', help='Specifies the used type of training.')
parser.add_argument('--folds', type=int, default=5, help='Specifies the number of folds.')
parser.add_argument('--meta_columns', type=int, nargs='+', default=[0, 1, 2], help='Used columns of meta data')
parser.add_argument('--multigraph', type=bool, default=False, help='Defines if multigraph setting is used')
parser.add_argument('--model', type=str, default='GAT', help='Specifies the used type of model.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
parser.add_argument('--patience', type=int, default=10, help='Patience until training is stopped with no improvement.')

parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--GPU_device', type=str, default='0', help='Specifies the used GPU.')
parser.add_argument('--writing', type=bool, default=False, help='Define if writer for Tensorboard should be used.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size.')
parser.add_argument('--label_list', type=str, nargs='+', default=[0, 1, 2], help='Used labels,'
                                                                                       'put as "--label_list "3" "5" .')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nfeat', type=int, default=20, help='Number of features used as input for GAT.')
parser.add_argument('--ngout', type=int, default=8, help='Dimension of GAT output.')
parser.add_argument('--nhid', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nheads', type=int, default=5, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.3, help='Alpha for the leaky_relu.')
parser.add_argument('-A', '--adj_type', type=int, default=1,
                    help='Adjacency metric type to use. 1 for thresholded '
                         'Gaussiankernel weighting')

args = parser.parse_args()

# --------------------------------------------------------
# Setup dataset folder
dirs = data_setup(args.dataset)
# --------------------------------------------------------

dataset_class = get_dataset_type(args.datatype)
dataset = dataset_class(dirs, args)


def get_mean(data):
    return np.round(np.mean(np.array(data), 0), 3)


def get_std(data):
    return np.round(np.std(np.array(data), 0), 3)


def metrics(preds, labels):
    if args.train_mode == 'classification':
        acc = np.sum(preds == labels) / (1.0 * np.size(test_labels))
        print('Accuracy:', acc)
        f1score = f1_score(labels, preds, average='macro')
        print('F1-Score:', f1score)
        sensitivity = np.sum(preds[preds == labels]) / np.sum(labels)
        print('Sensitivity:', sensitivity)
        specificity = (np.sum(preds == labels) - np.sum(preds[preds == labels])) / (np.size(test_labels) - np.sum(labels))
        print('Specificity:', specificity)
        print('---------------------------')
        print()
        return acc, f1score, sensitivity, specificity
    elif args.train_mode == 'regression':
        mae = np.sum(np.abs(preds - labels)) / (1.0 * np.size(labels))
        rmse = np.sqrt(np.sum(np.square(preds - labels)) / (1.0 * np.size(labels)))
        print()
        print('--------------------------------')
        print('Test metrics:')
        print('MAE:', mae)
        print('RMSE:', rmse)
        print('--------------------------------')
        return mae, rmse

if args.train_mode == 'classification':
    acc_list = []
    f1_score_list = []
    sensitivity_list = []
    specificity_list = []
else:
    mae_list = []
    rmse_list = []

feat_imp_list = []
phases = ['train', 'val', 'test']
for f in range(args.folds):
    dataset_sizes = {phase: dataset.dataset_size(f, phase) for phase in phases}
    dataloaders = {p: CustomDataLoader(dataset,
                                       batch_size=dataset_sizes[p],
                                       shuffle=False,
                                       num_workers=0, drop_last=False,
                                       sampler=dataset.get_sampler(f, p),
                                       phase=p) for p in phases}

    inputs_train, target_train, index_train, return_mode_train = next(iter(dataloaders['train']))
    inputs_val, target_val, index_val, return_mode_val = next(iter(dataloaders['val']))
    inputs_test, target_test, index_test, return_mode_test = next(iter(dataloaders['test']))

    train_data = np.concatenate((inputs_train['feat'].numpy(), inputs_val['feat'].numpy()), 0)
    train_labels = np.concatenate((target_train[0].numpy(), target_val[0].numpy()), 0)

    train_max = np.max(train_data, 0)
    train_min = np.min(train_data, 0)

    train_data -= train_min
    train_data /= (train_max - train_min)

    if args.train_mode == 'classification':
        clf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=None)
    else:
        clf = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=None)
    clf.fit(train_data, train_labels)

    test_data = inputs_test['feat'].numpy()
    test_data -= train_min
    test_data /= (train_max - train_min)

    test_labels = target_test[0].numpy()

    test_preds = clf.predict(test_data)

    print(test_preds)
    print(test_labels)
    if args.train_mode == 'classification':
        acc, f1score, sensitivity, specificity = metrics(test_preds, test_labels)
        acc_list.append(acc)
        f1_score_list.append(f1score)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
    else:
        mae, rmse = metrics(test_preds, test_labels)
        mae_list.append(mae)
        rmse_list.append(rmse)

    feature_importance = clf.feature_importances_
    feat_imp_list.append(feature_importance)

    # print(clf.feature_importances_)

if args.train_mode == 'classification':
    print('Overall accuracy is: {} +- {}'.format(get_mean(acc_list), get_std(acc_list)))
    print('Overall f1 score marco: {} +- {}'.format(get_mean(f1_score_list), get_std(f1_score_list)))
    print('Overall sensitivity: {} +- {}'.format(get_mean(sensitivity_list), get_std(sensitivity_list)))
    print('Overall specificity: {} +- {}'.format(get_mean(specificity_list), get_std(specificity_list)))
else:
    print('Overall mae is: {} +- {}'.format(get_mean(mae_list), get_std(mae_list)))
    print('Overall rmse marco: {} +- {}'.format(get_mean(rmse_list), get_std(rmse_list)))

feature_names = np.load('../data/{}/feature_names.npy'.format(args.dataset))
feat_imp_mean = get_mean(feat_imp_list)
feat_imp_std = get_std(feat_imp_list)
for i, name in enumerate(feature_names):
    print('The overall importance of {} is : {} +- {}.'.format(name, feat_imp_mean[i], feat_imp_std[i]))
