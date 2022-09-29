import os
import pickle
from typing import Any, Union

import numpy as np
import png
import pandas as pd

import torch
from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader
from sklearn.pipeline import Pipeline
from torchvision import datasets
from torchvision.datasets import MNIST
import urllib.request
import requests
import tarfile
import pandas as pd
import shutil
from sklearn import impute
from sklearn import preprocessing

from base.utils.configuration import Config
from base.utils import utils
from PIL import Image
import os.path

class DatasetSetupBuilder:

    def __init__(self):
        super().__init__()
        self.feat_block_list = []  # Specifies how features are grouped
        self.data_block_list = []
        self.dirs = None

    def build_dataset(self):
        """#TODO"""
        raise NotImplementedError

    """#TODO"""

    def setup_data(self):
        """#TODO"""
        self.build_dataset()
        Config.args['dirs'] = self.dirs

        self.get_feature_block_list()
        Config.args.dataset['feat_block_list'] = self.feat_block_list

    @staticmethod
    def create_dir(dir_):

        if not os.path.isdir(dir_):
            print('Create directory', dir_)
            os.mkdir(dir_)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        Config.global_dict['dataset'][name] = cls

    @staticmethod
    def get_class(config):
        """#TODO"""
        if config.args.dataset.name in config.global_dict.get('dataset'):

        	return Config.global_dict.get('dataset').get(config.args.dataset.name, None)
        else:
        	print(config.global_dict.keys())
        	raise KeyError('%s is not a known key.' % config.args.dataset.name)

    def get_feature_block_list(self):
        # self.input_ = """Get how features are grouped for every feature vector input.""" # changed by Alex

        feat_list = self.dirs.get('feat', None)
        if feat_list is not None:
            for k, v in enumerate(feat_list):
                len_feat_dim = np.load(v, allow_pickle=True).shape[-1]
                cur_feat_block = {str(i): [i] for i in range(len_feat_dim)}
                self.feat_block_list.append(cur_feat_block)


class MNIST5KImg(DatasetSetupBuilder):
    """#TODO"""

    def __init__(self):
        super(MNIST5KImg, self).__init__()

    def build_dataset(self):
        general_dir = "../data/MNIST5KImg/"
        data_dir = "../data/MNIST5KImg/MNIST_dowload/"
        folder_dir = "../data/MNIST5KImg/MNIST_folders/"

        if os.path.isdir(general_dir):
            print('MNIST folder structure already created, proceeding ...')
        else:
            print('Creating MNIST folder structure ...')
            self.create_dir('../data/')
            self.create_dir(general_dir)
            self.create_dir(data_dir)
            self.create_dir(general_dir + 'MNIST_numpy/')
            self.create_dir(folder_dir)
            self.create_dir(general_dir + 'meta_img/')
            for i in range(10):
                self.create_dir(folder_dir + '{}/'.format(i))

            mnist_train = MNIST(root=data_dir, train=True, download=True)  # 60000, 28, 28
            mnist_test = MNIST(root=data_dir, train=False, download=True)  # 10000, 28, 28

            mnist_data = torch.cat((mnist_train.data, mnist_test.data), 0)
            mnist_targets = torch.cat((mnist_train.targets, mnist_test.targets), 0)

            cur_data = []
            cur_targets = []
            for i in range(10):
                index = mnist_targets == i
                cur_data.append(mnist_data[index][:500])
                cur_targets.append(mnist_targets[index][:500])
            mnist_data = torch.cat(cur_data, 0)
            mnist_targets = torch.cat(cur_targets, 0)

            mnist_lower = mnist_data.numpy()[:, 14:28, :]
            mnist_higher = mnist_data.numpy()[:, 0:14, :]
            mnist_labels = mnist_targets.numpy()

            np.save(general_dir + 'MNIST_numpy/image_data.npy', mnist_lower)
            np.save(general_dir + 'MNIST_numpy/meta_data.npy', mnist_higher.reshape(-1, 14 * 28))
            np.save(general_dir + 'MNIST_numpy/targets.npy', mnist_labels)

            print('Creating MNIST images ...')
            img_paths = []
            for i in range(np.size(mnist_lower, 0)):
                img_path = folder_dir + '{}/Img_{}.png'.format(mnist_labels[i], i)
                png.from_array(mnist_lower[i, :, :], 'L').save(img_path)
                png.from_array(mnist_higher[i, :, :], 'L').save(general_dir + 'meta_img/Img_{}_m.png'.format(i))
                img_paths.append(img_path)
            np.save(general_dir + 'MNIST_numpy/img_paths.npy', np.array(img_paths))

        self.dirs = {'img': [folder_dir],
                     'img_paths': [general_dir + 'MNIST_numpy/img_paths.npy'],
                     'feat': None,
                     'seq': None,
                     'targets': [general_dir + 'MNIST_numpy/targets.npy'],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],
                     'meta': general_dir + 'MNIST_numpy/meta_data.npy',
                     'adj': general_dir + 'MNIST_numpy',
                     'adjtoinput': {'img': [0],
                                    'feat': None,
                                    'seq': None}
                     }


class MNIST5KImgExtract(DatasetSetupBuilder):
    """#TODO"""

    def __init__(self):
        super(MNIST5KImgExtract, self).__init__()

    def build_dataset(self):
        general_dir = "../data/MNIST5KImgExtract/"

        if os.path.isdir(general_dir):
            print('MNIST structure already created, proceeding ...')
        else:
            raise ValueError('MNIST extraction folder needs to get created first')

        features = []
        feat_adj = []
        for i in range(Config.args.folds):
            features.append(general_dir + 'extracted_features_f{}_b{}_l{}.npy'.format(i, Config.args.batch_size,
                                                                                      Config.args.label_list))
            feat_adj.append(0)
        self.dirs = {'img': None,
                     'img_paths': None,
                     'feat': features,
                     'seq': None,
                     'targets': [general_dir + 'targets_l{}.npy'.format(Config.args.label_list)],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],
                     'meta': general_dir + 'meta_data_l{}.npy'.format(Config.args.label_list),
                     'adj': general_dir + 'adj_l{}.npy'.format(Config.args.label_list),
                     'adjtoinput': {'img': None,
                                    'feat': feat_adj,
                                    'seq': None}
                     }


class MNIST20KImg(DatasetSetupBuilder):
    """#TODO"""

    def __init__(self):
        super(MNIST20KImg, self).__init__()

    def build_dataset(self):
        general_dir = "../data/MNIST20KImg/"
        data_dir = "../data/MNIST20KImg/MNIST_dowload/"
        folder_dir = "../data/MNIST20KImg/MNIST_folders/"

        if os.path.isdir(general_dir):
            print('MNIST folder structure already created, proceeding ...')
        else:
            print('Creating MNIST folder structure ...')
            self.create_dir('../data/')
            self.create_dir(general_dir)
            self.create_dir(data_dir)
            self.create_dir(general_dir + 'MNIST_numpy/')
            self.create_dir(folder_dir)
            self.create_dir(general_dir + 'meta_img/')
            for i in range(10):
                self.create_dir(folder_dir + '{}/'.format(i))

            mnist_train = MNIST(root=data_dir, train=True,
                                download=True)  # datasets.MNIST(root=data_dir, train=True, download=True)  # 60000, 28, 28
            mnist_test = MNIST(root=data_dir, train=False,
                               download=True)  # datasets.MNIST(root=data_dir, train=False, download=True)  # 10000, 28, 28

            mnist_data = torch.cat((mnist_train.data, mnist_test.data), 0)
            mnist_targets = torch.cat((mnist_train.targets, mnist_test.targets), 0)

            cur_data = []
            cur_targets = []
            for i in range(10):
                index = mnist_targets == i
                cur_data.append(mnist_data[index][:2000])
                cur_targets.append(mnist_targets[index][:2000])
            mnist_data = torch.cat(cur_data, 0)
            mnist_targets = torch.cat(cur_targets, 0)

            mnist_lower = mnist_data.numpy()[:, 14:28, :]
            mnist_higher = mnist_data.numpy()[:, 0:14, :]
            mnist_labels = mnist_targets.numpy()

            np.save(general_dir + 'MNIST_numpy/image_data.npy', mnist_lower)
            np.save(general_dir + 'MNIST_numpy/meta_data.npy', mnist_higher.reshape(-1, 14 * 28))
            np.save(general_dir + 'MNIST_numpy/targets.npy', mnist_labels)

            print('Creating MNIST images ...')
            img_paths = []
            for i in range(np.size(mnist_lower, 0)):
                img_path = folder_dir + '{}/Img_{}.png'.format(mnist_labels[i], i)
                png.from_array(mnist_lower[i, :, :], 'L').save(img_path)
                png.from_array(mnist_higher[i, :, :], 'L').save(general_dir + 'meta_img/Img_{}_m.png'.format(i))
                img_paths.append(img_path)
            np.save(general_dir + 'MNIST_numpy/img_paths.npy', np.array(img_paths))

        self.dirs = {'img': [folder_dir],
                     'img_paths': [general_dir + 'MNIST_numpy/img_paths.npy'],
                     'feat': None,
                     'seq': None,
                     'targets': [general_dir + 'MNIST_numpy/targets.npy'],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],
                     'meta': general_dir + 'MNIST_numpy/meta_data.npy',
                     'adj': general_dir + 'MNIST_numpy',
                     'adjtoinput': {'img': [0],
                                    'feat': None,
                                    'seq': None}
                     }


class MNIST20KImgExtract(DatasetSetupBuilder):
    """#TODO"""

    def __init__(self):
        super(MNIST20KImgExtract, self).__init__()

    def build_dataset(self):
        general_dir = "../data/MNIST20KImgExtract/"

        if os.path.isdir(general_dir):
            print('MNIST structure already created, proceeding ...')
        else:
            raise ValueError('MNIST extraction folder needs to get created first')

        features = []
        feat_adj = []
        for i in range(Config.args.folds):
            features.append(general_dir + 'extracted_features_f{}_b{}_l{}.npy'.format(i, Config.args.batch_size,
                                                                                      Config.args.label_list))
            feat_adj.append(0)
        self.dirs = {'img': None,
                     'img_paths': None,
                     'feat': features,
                     'seq': None,
                     'targets': [general_dir + 'targets_l{}.npy'.format(Config.args.label_list)],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],
                     'meta': general_dir + 'meta_data_l{}.npy'.format(Config.args.label_list),
                     'adj': general_dir + 'adj_l{}.npy'.format(Config.args.label_list),
                     'adjtoinput': {'img': None,
                                    'feat': feat_adj,
                                    'seq': None}
                     }


class MNIST5KFeat(DatasetSetupBuilder):
    """#TODO MNIST dataset where the lower part of the image is used as feature-based,
        and the upper part is used as meta-data."""

    def __init__(self):
        super(MNIST5KFeat, self).__init__()

    def build_dataset(self):

        general_dir = "../data/MNISTFeat/"
        data_dir = "../data/MNISTFeat/MNIST_dowload/"

        if os.path.isdir(general_dir):
            print('MNISTFeat folder structure already created, proceeding ...')
        else:
            print('Creating MNISTFeat folder structure ...')
            self.create_dir('../data/')
            self.create_dir(general_dir)
            self.create_dir(data_dir)
            self.create_dir(general_dir + 'MNIST_numpy/')

            mnist_train = MNIST(root=data_dir, train=True,
                                download=True)  # datasets.MNIST(root=data_dir, train=True, download=True)  # 60000, 28, 28
            mnist_test = MNIST(root=data_dir, train=False,
                               download=True)  # datasets.MNIST(root=data_dir, train=False, download=True)  # 10000, 28, 28
            mnist_data = torch.cat((mnist_train.data, mnist_test.data), 0)
            mnist_targets = torch.cat((mnist_train.targets, mnist_test.targets), 0)
            cur_data = []
            cur_targets = []
            for i in range(10):
                index = mnist_targets == i
                cur_data.append(mnist_data[index][:500])
                cur_targets.append(mnist_targets[index][:500])
            mnist_data = torch.cat(cur_data, 0)
            mnist_targets = torch.cat(cur_targets, 0)

            mnist_lower = mnist_data.numpy()[:, 14:28, :].reshape(-1, 14 * 28)
            mnist_lower = mnist_lower / 255.
            mnist_higher = mnist_data.numpy()[:, 0:14, :].reshape(-1, 14 * 28)
            mnist_labels = mnist_targets.numpy()

            np.save(general_dir + 'MNIST_numpy/features.npy', mnist_lower)
            np.save(general_dir + 'MNIST_numpy/meta_data.npy', mnist_higher)
            np.save(general_dir + 'MNIST_numpy/targets.npy', mnist_labels)

            print('MNIST folder structure created, proceeding ...')

        self.dirs = {'img': None,
                     'img_paths': None,
                     'feat': [general_dir + 'MNIST_numpy/features.npy'],
                     # 'feat': [general_dir + 'MNIST_numpy/features.npy', general_dir + 'MNIST_numpy/features.npy'],
                     # [general_dir + 'MNIST_numpy/features.npy', general_dir + 'MNIST_numpy/features.npy'],
                     'seq': None,
                     'targets': [general_dir + 'MNIST_numpy/targets.npy'],
                     # [general_dir + 'MNIST_numpy/targets.npy', general_dir + 'MNIST_numpy/targets.npy'],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],  # ['CrossEntropyLoss', 'CrossEntropyLoss'],
                     'meta': general_dir + 'MNIST_numpy/meta_data.npy',
                     'adj': general_dir + 'MNIST_numpy',
                     'adjtoinput': {'img': None,
                                    'feat': [0],  # [0, 1],
                                    'seq': None}
                     }


class MNIST5KFeatImg(DatasetSetupBuilder):
    """#TODO"""

    def __init__(self):
        super(MNIST5KFeatImg, self).__init__()

    def build_dataset(self):
        general_dir = "../data/MNISTFeatImg/"
        data_dir = "../data/MNISTFeatImg/MNIST_dowload/"
        folder_dir = "../data/MNISTFeatImg/MNIST_folders/"

        if os.path.isdir(general_dir):
            print('MNIST folder structure already created, proceeding ...')
        else:
            print('Creating MNIST folder structure ...')
            self.create_dir('../data/')
            self.create_dir(general_dir)
            self.create_dir(data_dir)
            self.create_dir(general_dir + 'MNIST_numpy/')
            self.create_dir(folder_dir)
            self.create_dir(general_dir + 'meta_img/')
            for i in range(10):
                self.create_dir(folder_dir + '{}/'.format(i))

            mnist_train = MNIST(root=data_dir, train=True,
                                download=True)  # 60000, 28, 28
            mnist_test = MNIST(root=data_dir, train=False,
                               download=True)  # 10000, 28, 28

            mnist_data = torch.cat((mnist_train.data, mnist_test.data), 0)
            mnist_targets = torch.cat((mnist_train.targets, mnist_test.targets), 0)
            cur_data = []
            cur_targets = []
            for i in range(10):
                index = mnist_targets == i
                cur_data.append(mnist_data[index][:500])
                cur_targets.append(mnist_targets[index][:500])
            mnist_data = torch.cat(cur_data, 0)
            mnist_targets = torch.cat(cur_targets, 0)

            mnist_lower_l = mnist_data.numpy()[:, 14:28, :14]
            mnist_lower_r = mnist_data.numpy()[:, 14:28, 14:].reshape(-1, 14 * 14)
            mnist_lower_r = mnist_lower_r / 255.0
            mnist_higher = mnist_data.numpy()[:, 0:14, :].reshape(-1, 14 * 28)
            mnist_labels = mnist_targets.numpy()

            np.save(general_dir + 'MNIST_numpy/features.npy', mnist_lower_r)
            np.save(general_dir + 'MNIST_numpy/meta_data.npy', mnist_higher)
            np.save(general_dir + 'MNIST_numpy/targets.npy', mnist_labels)

            print('Creating MNIST images ...')
            img_paths = []
            for i in range(np.size(mnist_lower_l, 0)):
                img_path = folder_dir + '{}/Img_{}.png'.format(mnist_labels[i], i)
                png.from_array(mnist_lower_l[i, :, :], 'L').save(img_path)
                img_paths.append(img_path)
            np.save(general_dir + 'MNIST_numpy/img_paths.npy', np.array(img_paths))

            print('MNIST folder structure created, proceeding ...')
        self.dirs = {'img': [folder_dir, folder_dir],
                     'img_paths': [general_dir + 'MNIST_numpy/img_paths.npy',
                                   general_dir + 'MNIST_numpy/img_paths.npy'],
                     'feat': [general_dir + 'MNIST_numpy/features.npy', general_dir + 'MNIST_numpy/features.npy'],
                     'seq': None,
                     'targets': [general_dir + 'MNIST_numpy/targets.npy', general_dir + 'MNIST_numpy/targets.npy'],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 0.5}, {'loss': 'CrossEntropyLoss', 'wts': 0.5}],
                     'meta': general_dir + 'MNIST_numpy/meta_data.npy',
                     'adj': general_dir + 'MNIST_numpy',
                     'adjtoinput': {'img': [0, 1],
                                    'feat': [0, 1],
                                    'seq': None}
                     }


class MNIST5KFeatImgSeq(DatasetSetupBuilder):
    """#TODO"""

    def __init__(self):
        super(MNIST5KFeatImgSeq, self).__init__()

    def build_dataset(self):
        general_dir = "../data/MNIST5KFeatImgSeq/"
        data_dir = "../data/MNIST5KFeatImgSeq/MNIST_dowload/"
        folder_dir = "../data/MNIST5KFeatImgSeq/MNIST_folders/"

        if os.path.isdir(general_dir):
            print('MNIST folder structure already created, proceeding ...')
        else:
            print('Creating MNIST folder structure ...')
            self.create_dir('../data/')
            self.create_dir(general_dir)
            self.create_dir(data_dir)
            self.create_dir(general_dir + 'MNIST_numpy/')
            self.create_dir(folder_dir)
            self.create_dir(general_dir + 'meta_img/')
            for i in range(10):
                self.create_dir(folder_dir + '{}/'.format(i))

            mnist_train = MNIST(root=data_dir, train=True,
                                download=True)  # 60000, 28, 28
            mnist_test = MNIST(root=data_dir, train=False, download=True)  # 10000, 28, 28

            mnist_data = torch.cat((mnist_train.data, mnist_test.data), 0)
            mnist_targets = torch.cat((mnist_train.targets, mnist_test.targets), 0)
            cur_data = []
            cur_targets = []
            for i in range(10):
                index = mnist_targets == i
                cur_data.append(mnist_data[index][:500])
                cur_targets.append(mnist_targets[index][:500])
            mnist_data = torch.cat(cur_data, 0)
            mnist_targets = torch.cat(cur_targets, 0)

            mnist_lower_l = mnist_data.numpy()[:, 14:28, :14]
            mnist_lower_r = mnist_data.numpy()[:, 14:28, 14:].reshape(-1, 14 * 14) / 255.0
            mnist_higher_l = mnist_data.numpy()[:, 0:14, :14].reshape(-1, 14 * 14)
            mnist_higher_r = mnist_data.numpy()[:, 0:14, 14:].reshape(-1, 14, 14) / 255.0
            mnist_labels = mnist_targets.numpy()

            np.save(general_dir + 'MNIST_numpy/features.npy', mnist_lower_r)
            np.save(general_dir + 'MNIST_numpy/meta_data.npy', mnist_higher_l)
            np.save(general_dir + 'MNIST_numpy/seq_data.npy', mnist_higher_r)
            np.save(general_dir + 'MNIST_numpy/targets.npy', mnist_labels)

            print('Creating MNIST images ...')
            img_paths = []
            for i in range(np.size(mnist_lower_l, 0)):
                img_path = folder_dir + '{}/Img_{}.png'.format(mnist_labels[i], i)
                png.from_array(mnist_lower_l[i, :, :], 'L').save(img_path)
                img_paths.append(img_path)
            np.save(general_dir + 'MNIST_numpy/img_paths.npy', np.array(img_paths))

            print('MNIST folder structure created, proceeding ...')
        self.dirs = {'img': [folder_dir],
                     'img_paths': [general_dir + 'MNIST_numpy/img_paths.npy'],
                     'feat': [general_dir + 'MNIST_numpy/features.npy'],
                     'seq': [general_dir + 'MNIST_numpy/seq_data.npy'],
                     'targets': [general_dir + 'MNIST_numpy/targets.npy'],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],
                     'meta': general_dir + 'MNIST_numpy/meta_data.npy',
                     'adj': general_dir + 'MNIST_numpy',
                     'adjtoinput': {'img': [0],
                                    'feat': [0],
                                    'seq': [0]}
                     }


class MNIST1KFeatImgSeq(DatasetSetupBuilder):
    """#TODO"""

    def __init__(self):
        super(MNIST1KFeatImgSeq, self).__init__()

    def build_dataset(self):
        general_dir = "../data/MNIST1KFeatImgSeq/"
        data_dir = "../data/MNIST1KFeatImgSeq/MNIST_dowload/"
        folder_dir = "../data/MNIST1KFeatImgSeq/MNIST_folders/"

        if os.path.isdir(general_dir):
            print('MNIST folder structure already created, proceeding ...')
        else:
            print('Creating MNIST folder structure ...')
            self.create_dir('../data/')
            self.create_dir(general_dir)
            self.create_dir(data_dir)
            self.create_dir(general_dir + 'MNIST_numpy/')
            self.create_dir(folder_dir)
            self.create_dir(general_dir + 'meta_img/')
            for i in range(10):
                self.create_dir(folder_dir + '{}/'.format(i))

            mnist_train = MNIST(root=data_dir, train=True,
                                download=True)  # 60000, 28, 28
            mnist_test = MNIST(root=data_dir, train=False,
                               download=True)  # 10000, 28, 28

            mnist_data = torch.cat((mnist_train.data, mnist_test.data), 0)
            mnist_targets = torch.cat((mnist_train.targets, mnist_test.targets), 0)
            cur_data = []
            cur_targets = []
            for i in range(10):
                index = mnist_targets == i
                cur_data.append(mnist_data[index][:100])
                cur_targets.append(mnist_targets[index][:100])
            mnist_data = torch.cat(cur_data, 0)
            mnist_targets = torch.cat(cur_targets, 0)

            mnist_lower_l = mnist_data.numpy()[:, 14:28, :14]
            mnist_lower_r = mnist_data.numpy()[:, 14:28, 14:].reshape(-1, 14 * 14) / 255.0
            mnist_higher_l = mnist_data.numpy()[:, 0:14, :14].reshape(-1, 14 * 14)
            mnist_higher_r = mnist_data.numpy()[:, 0:14, 14:].reshape(-1, 14, 14) / 255.0
            mnist_labels = mnist_targets.numpy()

            np.save(general_dir + 'MNIST_numpy/features.npy', mnist_lower_r)
            np.save(general_dir + 'MNIST_numpy/meta_data.npy', mnist_higher_l)
            np.save(general_dir + 'MNIST_numpy/seq_data.npy', mnist_higher_r)
            np.save(general_dir + 'MNIST_numpy/targets.npy', mnist_labels)

            print('Creating MNIST images ...')
            img_paths = []
            for i in range(np.size(mnist_lower_l, 0)):
                img_path = folder_dir + '{}/Img_{}.png'.format(mnist_labels[i], i)
                png.from_array(mnist_lower_l[i, :, :], 'L').save(img_path)
                img_paths.append(img_path)
            np.save(general_dir + 'MNIST_numpy/img_paths.npy', np.array(img_paths))

            print('MNIST folder structure created, proceeding ...')
        self.dirs = {'img': [folder_dir],
                     'img_paths': [general_dir + 'MNIST_numpy/img_paths.npy'],
                     'feat': [general_dir + 'MNIST_numpy/features.npy'],
                     'seq': [general_dir + 'MNIST_numpy/seq_data.npy'],
                     'targets': [general_dir + 'MNIST_numpy/targets.npy'],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],
                     'meta': general_dir + 'MNIST_numpy/meta_data.npy',
                     'adj': general_dir + 'MNIST_numpy',
                     'adjtoinput': {'img': [0],
                                    'feat': [0],
                                    'seq': [0]}
                     }


class ToysetFeat(DatasetSetupBuilder):
    """#TODO"""

    def __init__(self):
        super(ToysetFeat, self).__init__()

    def build_dataset(self):
        if not os.path.isdir('../data/ToySet'):
            print('Creating ToySet folder structure ...')
            toy_num = 1000
            features = np.random.rand(toy_num, 20)
            targets = np.random.randint(0, 3, toy_num)
            age = np.random.randint(0, 40, toy_num)
            gender = np.random.randint(0, 2, toy_num)
            weight = np.random.randint(50, 70, toy_num)
            age += 20 * (targets + 1)
            weight += 10 * (targets + 1)
            meta_data = np.array([age, gender, weight]).transpose()

            self.create_dir('../data')
            self.create_dir('../data/ToySet')
            np.save('../data/ToySet/features.npy', features)
            np.save('../data/ToySet/targets.npy', targets)
            np.save('../data/ToySet/meta_data.npy', meta_data)
            print('ToySet folder structure created, proceeding ...')
        else:
            print('ToySet folder structure already created, proceeding ...')

        self.dirs = {'img': None,
                     'feat': '../data/ToySet',
                     'targets': ['../data/ToySet'],
                     'meta': '../data/ToySet',
                     'loss': [torch.nn.CrossEntropyLoss],
                     'adj': '../data/'}


class ChestXray14SingleLabel(DatasetSetupBuilder):
    def __init__(self):
        super(ChestXray14SingleLabel, self).__init__()

    def build_dataset(self):
        """
        Creates ChestXray14 dataset
        """

        general_dir = '../data/ChestXray14SL'
        folder_dir = '../data/ChestXray14SL/ChestXray14SL_folders'
        data_dir = '../data/ChestXray14SL/ChestXray14SL_numpy'

        if not os.path.exists('{}/Data_Entry_2017_v2020.csv'.format(general_dir)):
            raise AssertionError('Please store the file Data_Entry_2017_v2020.csv from the NIH database'
                                 'https://nihcc.app.box.com/v/ChestXray-NIHCC/file/219760887468'
                                 'within the folder /data/ChestXray14SL before proceeding.')

        if False:  # os.path.isdir(folder_dir):
            print('ChestX-ray 14 folder structure already created, proceeding ...')
        else:
            print('Creating ChestX-ray 14 folder structure ...')

            self.create_dir('../data')
            self.create_dir(general_dir)
            self.create_dir(folder_dir)
            self.create_dir(data_dir)

            links = [
                'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz'  # ,
                'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz'  # ,
                'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
                # 'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
                # 'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
                # 'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
                # 'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
                # 'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
                # 'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
                # 'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
                # 'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
                # 'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
            ]
            print('Downloading ChestX-ray 14 images, dataset size > 40 GB in total ...')
            print('Depending on your internet connection the download might take several minutes ...')

            if not os.path.isdir(general_dir + '/images'):
                for i, url in enumerate(links):
                    thetarfile = url
                    print('Downloading ChestX-ray 14 images part {} from {} ...'.format(i + 1, url))
                    ftpstream = urllib.request.urlopen(thetarfile)
                    thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
                    thetarfile.extractall(general_dir)
                print('All ChestX-ray 14 images successfully downloaded')
            else:
                print('Images already loaded, proceeding ...')
            num_img = len([name for name in os.listdir(general_dir + '/images')])
            print('Total amount of images:', num_img)

            df = pd.read_csv('{}/Data_Entry_2017_v2020.csv'.format(general_dir), sep=',')
            df = df.head(num_img)

            labels = np.array(list(df['Finding Labels'].values))
            label_list = np.unique(labels)

            print('Single labels found for training:')
            print('---------------')
            single_label_list = []
            for label in list(label_list):
                if '|' not in label:
                    single_label_list.append(label)
                    print(label)
            print('---------------')
            print()

            for label in single_label_list:
                self.create_dir('{}/{}'.format(folder_dir, label))

            print()
            print('Moving images into corresponding class folders...')
            ImgPath_list = []
            targets_list = []
            PatID_list = []
            age_list = []
            gender_list = []
            position_list = []

            for idx, img_path in enumerate(df['Image Index'].values):
                label = df['Finding Labels'].values[idx]
                if label in single_label_list:
                    if os.path.exists('{}/images/{}'.format(general_dir, img_path)):
                        shutil.move('{}/images/{}'.format(general_dir, img_path),
                                    '{}/{}/{}'.format(folder_dir, label, img_path))
                    ImgPath_list.append('{}/{}/{}'.format(folder_dir, label, img_path))
                    targets_list.append(single_label_list.index(label))
                    PatID_list.append(df['Patient ID'].values[idx])  # TODO use PatID list to put same Pat in same fold
                    age_list.append(df['Patient Age'].values[idx])
                    gender_list.append(int(df['Patient Gender'].values[idx] == 'M'))
                    position_list.append(int(df['View Position'].values[idx] == 'PA'))
            print('Images successfully moved, deleting multi-label images ...')
            shutil.rmtree(general_dir + '/images')
            print('Multi-label images deleted')
            print()

            features = np.array([PatID_list, age_list, gender_list, position_list])
            meta_data = np.array([PatID_list, age_list, gender_list, position_list])

            print('Storing additional data ...')
            np.save('{}/img_paths.npy'.format(data_dir), np.array(ImgPath_list))
            np.save('{}/features.npy'.format(data_dir), np.transpose(features))
            np.save('{}/targets.npy'.format(data_dir), np.array(targets_list))
            np.save('{}/meta_data.npy'.format(data_dir), np.transpose(meta_data))
            print('All data stored, dataset created')

        self.dirs = {'img': [folder_dir, folder_dir],
                     'img_paths': [data_dir + '/img_paths.npy', data_dir + '/img_paths.npy'],
                     'feat': [data_dir + '/features.npy', data_dir + '/features.npy'],
                     'seq': None,
                     'targets': [data_dir + '/targets.npy'],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],
                     'meta': data_dir + '/meta_data.npy',
                     'adj': data_dir,
                     'adjtoinput': {'img': [0, 1],
                                    'feat': [0, 1],
                                    'seq': None}
                     }


class TADPOLEMeanImputed(DatasetSetupBuilder):
    """Mean-imputed, mean-centered, and unit variance scaled TADPOLE dataset.
    # TODO
    """

    def __init__(self):
        super(TADPOLEMeanImputed, self).__init__()

    def build_dataset(self):
        general_dir = "../data/tadpoleMeanImputed/"
        self.create_dir(general_dir)
        feat_data_path = os.path.join(general_dir, 'features.npy')
        meta_data_path = os.path.join(general_dir, 'meta_data.npy')
        target_data_path = os.path.join(general_dir, 'targets.npy')

        if not os.path.exists(feat_data_path) or not os.path.exists(meta_data_path) or not os.path.exists(
                target_data_path):
            meta_data = pd.read_csv('../data/tadpole/adni_one_baseline_meta_data.csv')
            target_data = pd.read_csv('../data/tadpole/adni_one_baseline_label_data.csv')
            target_data = target_data.squeeze()
            feat_data = pd.read_csv('../data/tadpole/adni_one_baseline_feature_data.csv')

            print("meta data dimension is: {}".format(meta_data.shape))
            print("x data dimension is: {}".format(feat_data.shape))
            print("target_data data dimension is: {}".format(target_data.shape))

            meta_data.PTGENDER = meta_data.PTGENDER.replace('Male', 0)
            meta_data.PTGENDER = meta_data.PTGENDER.replace('Female', 1)

            print('Feature shape is {}'.format(feat_data.shape))
            print('Class label shape is {}'.format(target_data.shape))
            print('Meta information shape is {}'.format(meta_data.shape))

            feat_data.replace(-4, np.nan, inplace=True)
            feat_data.replace('<200', np.nan, inplace=True)
            feat_data.replace('^\s*$', np.nan, inplace=True, regex=True)
            feat_data[feat_data.applymap(type) == str] = np.nan

            feat_data = utils.df_drop(feat_data, .10)[0]

            # Define blocks
            # mri_block_cols = feat_data.columns.str.contains('UCSFFSX')
            # pet_block_cols = feat_data.columns.str.contains('BAIPETNMRC|UCBERKELEYAV45|UCBERKELEYAV1451')
            # dti_block_cols = feat_data.columns.str.contains('DTIROI')
            # csf_block_cols = feat_data.columns.str.contains('UPENNBIOM')

            # Impute using mean
            # imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
            # normalize = preprocessing.StandardScaler()
            # pipe = Pipeline([('imputer', imp), ('normalizer', normalize)])
            # feat_data = pipe.fit_transform(np.array(feat_data))

            print('Initializing TADPOLE dataset...')
            np.save(feat_data_path, feat_data)
            np.save(meta_data_path, meta_data)
            np.save(target_data_path, target_data)

        else:
            print('Loading saved TADPOLE dataset...')

        self.dirs = {'img': None,
                     'img_paths': None,
                     'feat': [feat_data_path, feat_data_path, feat_data_path],
                     'seq': None,
                     'targets': [target_data_path],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],
                     'meta': meta_data_path,
                     'adj': general_dir,
                     'adjtoinput': {'img': None,
                                    'feat': [0, 1, 2],
                                    'seq': None}
                     }


class TADPOLE(DatasetSetupBuilder):
    """Pre-process TADPOLE dataset similar to MGMC which contains missing features."""

    def __init__(self):
        super(TADPOLE, self).__init__()

    def build_dataset(self):
        general_dir = "../data/tadpole/"
        self.create_dir(general_dir)
        feat_data_path = os.path.join(general_dir, 'features.npy')
        meta_data_path = os.path.join(general_dir, 'meta_data.npy')
        target_data_path = os.path.join(general_dir, 'targets.npy')

        if not os.path.exists(feat_data_path) or not os.path.exists(meta_data_path) or not os.path.exists(
                target_data_path):
            meta_data = pd.read_csv('../data/tadpole/adni_one_baseline_meta_data_missing.csv')[
                ['AGE', 'PTGENDER', 'APOE4']]
            target_data = pd.read_csv('../data/tadpole/adni_one_baseline_label_data.csv')
            target_data = target_data.squeeze()
            feat_data = pd.read_csv('../data/tadpole/adni_one_baseline_feature_data_missing.csv')

            feat_data.replace(-4, np.nan, inplace=True)
            feat_data.replace('<200', np.nan, inplace=True)
            feat_data.replace('^\s*$', np.nan, inplace=True, regex=True)
            feat_data[feat_data.applymap(type) == str] = np.nan

            # Pre-process meta information
            meta_data.PTGENDER = meta_data.PTGENDER.replace('Male', 0)
            meta_data.PTGENDER = meta_data.PTGENDER.replace('Female', 1)

            # One-hot encode APOE4 before concatenating meta data to the feature matrix
            apoe_data = np.eye(meta_data.APOE4.unique().size)[np.array(meta_data.APOE4).astype(int)]
            concat_meta = np.concatenate([np.array(meta_data[['AGE', 'PTGENDER']]), apoe_data], 1)
            feat_data = np.concatenate([feat_data, meta_data], 1).astype(np.float32)

            print('Feature shape is {}'.format(feat_data.shape))
            print('Class label shape is {}'.format(target_data.shape))
            print('Meta information shape is {}'.format(meta_data.shape))

            print('Initializing TADPOLE dataset...')
            np.save(feat_data_path, feat_data)
            np.save(meta_data_path, meta_data)
            np.save(target_data_path, target_data)

        else:
            print('Loading saved TADPOLE dataset...')

        self.dirs = {'img': None,
                     'img_paths': None,
                     'feat': [feat_data_path],
                     'seq': None,
                     'targets': [target_data_path],
                     # 'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],
                     'meta': meta_data_path,
                     'adj': general_dir,
                     'adjtoinput': {'img': None,
                                    'feat': [0],
                                    'seq': None}
                     }


class TADPOLESplitted(DatasetSetupBuilder):
    """Pre-process TADPOLE dataset similar to MGMC which contains missing features."""

    def __init__(self):
        super(TADPOLESplitted, self).__init__()

    def build_dataset(self):
        general_dir = "../data/tadpoleSplitted/"
        self.create_dir(general_dir)
        feat_csf_path = os.path.join(general_dir, 'features_csf.npy')
        feat_mri_path = os.path.join(general_dir, 'features_mri.npy')
        feat_pet_path = os.path.join(general_dir, 'features_pet.npy')
        meta_data_path = os.path.join(general_dir, 'meta_data.npy')
        meta_data_as_input_path = os.path.join(general_dir, 'meta_data_as_input.npy')
        target_data_path = os.path.join(general_dir, 'targets.npy')
        path_list = [feat_csf_path, feat_mri_path, feat_pet_path, meta_data_path, meta_data_as_input_path,
                     target_data_path]

        # If at least one of the files is missing, just build all files again
        build_flag = 0
        for i in path_list:
            if build_flag == 0 and not os.path.exists(i):
                build_flag = 1

        if build_flag:
            print('Initializing Splitted TADPOLE dataset...')
            meta_data = pd.read_csv('../data/tadpoleSplitted/adni_one_baseline_meta_data.csv')
            meta_data = meta_data[['AGE', 'PTGENDER', 'APOE4']]
            target_data = pd.read_csv('../data/tadpoleSplitted/adni_one_baseline_label_data_raw.csv')
            target_data = target_data.squeeze()
            # 1 dimensional
            feat_csf_data = pd.read_csv('../data/tadpoleSplitted/adni_one_feat_csf.csv')
            # 328 dimensional
            feat_mri_data = pd.read_csv('../data/tadpoleSplitted/adni_one_feat_mri.csv')
            # 106 dimensional
            feat_pet_data = pd.read_csv('../data/tadpoleSplitted/adni_one_feat_pet.csv')

            meta_data.PTGENDER = meta_data.PTGENDER.replace('Male', 0)
            meta_data.PTGENDER = meta_data.PTGENDER.replace('Female', 1)

            # Convert APOE to one hot encoding
            num_unique_elem = len(meta_data.APOE4.unique())
            encoded_apoe4 = np.eye(num_unique_elem)[np.array(meta_data.APOE4).astype(int)]

            # Save separately meta information which will be used as input and used to calculate graph
            meta_data_as_input = np.array(meta_data.iloc[:, :2])
            meta_data_as_input = np.concatenate([meta_data_as_input, encoded_apoe4], 1)
            meta_data = np.array(meta_data)

            feat_csf_data = np.array(feat_csf_data).astype(np.float32)
            feat_mri_data = np.array(feat_mri_data).astype(np.float32)
            feat_pet_data = np.array(feat_pet_data).astype(np.float32)
            target_data = np.array(target_data)

            print("mri data dimension is: {}".format(feat_mri_data.shape))
            print("pet data dimension is: {}".format(feat_pet_data.shape))
            print("csf data dimension is: {}".format(feat_csf_data.shape))
            print("meta data as feture input dimension is: {}".format(meta_data_as_input.shape))
            print("meta data dimension is: {}".format(meta_data.shape))
            print("target data dimension is: {}".format(target_data.shape))
            print('Saving dataset setup numpy arrays...')
            np.save(feat_csf_path, feat_csf_data)
            np.save(feat_mri_path, feat_mri_data)
            np.save(feat_pet_path, feat_pet_data)
            np.save(meta_data_as_input_path, meta_data_as_input)
            np.save(meta_data_path, meta_data)
            np.save(target_data_path, target_data)
            print('TADPOLE numpy arrays saved...')

        else:
            print('Loading saved Splitted TADPOLE dataset...')

        self.dirs = {'img': None,
                     'img_paths': None,
                     'feat': [feat_mri_path, feat_pet_path, feat_csf_path, meta_data_as_input_path],
                     'seq': None,
                     'targets': [target_data_path],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],
                     'meta': meta_data_path,
                     'adj': general_dir,
                     'adjtoinput': {'img': None,
                                    'feat': [0, 1, 2, 3],
                                    'seq': None}
                     }


class UCIThyroid(DatasetSetupBuilder):
    """
    Thyroid dataset loader for for Thyroid disease classification.
    (https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/)
    """

    def __init__(self):
        super(UCIThyroid, self).__init__()

    def build_dataset(self):
        super().__init__()

        general_dir = "../data/thyroid/"
        feat_data_path = os.path.join(general_dir, 'features.npy')
        meta_data_path = os.path.join(general_dir, 'meta_data.npy')
        target_data_path = os.path.join(general_dir, 'targets.npy')

        if not os.path.exists(feat_data_path) or not os.path.exists(meta_data_path) or not os.path.exists(
                target_data_path):

            # Column names in dataframe
            # col_names_in_df = ['age',  # continuous
            #          'sex',  # binary 0/1
            #          'on thyroxine',  # binary 0/1
            #          'query on thyroxine',  # binary 0/1
            #          'on antithyroid medication',  # binary 0/1
            #          'sick',  # binary 0/1
            #          'pregnant',  # binary 0/1
            #          'thyroid surgery',  # binary 0/1
            #          'I131 treatment',  # binary 0/1
            #          'query hypothyroid',  # binary 0/1
            #          'query hyperthyroid',  # binary 0/1
            #          'lithium',  # binary 0/1
            #          'goitre',  # binary 0/1
            #          'tumor',  # binary 0/1
            #          'hypopituitary',  # binary 0/1
            #          'psych',  # binary 0/1
            #          'TSH',  # continuous
            #          'T3',  # continuous
            #          'TT4',  # continuous
            #          'T4U',  # continuous
            #          'FTI1',  # continuous
            #          'diagnoses']  # 3 classes: 1, 2, 3

            tr_csv_path = '../data/thyroid/ann-train.csv'
            ts_csv_path = '../data/thyroid/ann-test.csv'
            tr_data = pd.read_csv(tr_csv_path, header=None)
            ts_data = pd.read_csv(ts_csv_path, header=None)
            data_df = pd.concat([tr_data, ts_data], 0)
            data_df = np.array(data_df)

            feature_data = data_df[:, :-1]  # Use only all continuous variables
            meta_data = feature_data[:, :-5]
            feature_data = feature_data[:, -5:]
            target_data = data_df[:, -1]
            target_data = target_data - 1  # Change start class label to 0

            print("meta data dimension is: {}".format(meta_data.shape))
            print("x data dimension is: {}".format(feature_data.shape))
            print("target_data data dimension is: {}".format(target_data.shape))

            print('Initializing UCIThyroid dataset...')
            np.save(feat_data_path, feature_data)
            np.save(meta_data_path, meta_data)
            np.save(target_data_path, target_data)

        else:
            print('Loading saved UCIThyroid dataset...')

        self.dirs = {'img': None,
                     'img_paths': None,
                     'feat': [feat_data_path],
                     'seq': None,
                     'targets': [target_data_path],
                     'loss': [{'loss': 'CrossEntropyLoss', 'wts': 1.0}],
                     'meta': meta_data_path,
                     'adj': general_dir,
                     'adjtoinput': {'img': None,
                                    'feat': [0],
                                    'seq': None}
                     }


class COVIDiCTCF(DatasetSetupBuilder):
    """#TODO"""

    def __init__(self):
        super(COVIDiCTCF, self).__init__()

    def build_dataset(self):
        general_dir = "../data/covid/"
        image_dir = "../data/covid/images/"
        # image_dir = "/home/ubuntu/local-s3-bucket/cohort1/"  # "../data/covid/images/"
        # image_dir = "../data/covid/images/mosaic/"  # "../data/covid/images/"
        # data_dir = "../data/covid/covid_download/"

        print('Initializing COVID dataset...')

        if False:  # os.path.isdir(general_dir): #DEBUG
            print('covid folder structure already created, proceeding ...')
        else:
            print('Creating covid folder structure ...')
            self.create_dir('../data/')
            self.create_dir(general_dir)

            SCALE_METHOD = "Standardization"  # decide between "MinMax" and "Standardization"

            # paths to input data
            meta_path = "../data/covid/covid_download/CF_meta.csv"
            overview_path = "../data/covid/covid_download/CF_overview.csv"
            # feat_path = "../data/covid/covid_download/CF_clinical_features.csv"
            feat_path = "../data/covid/covid_download/iCTCF_CF_features.csv"
            label_path = "../data/covid/covid_download/CF_labels_all.csv"

            image_path_list = [image_dir + 'mosaic/' + name for name in os.listdir(image_dir + 'mosaic/')]
            num_images_list = []

            for i in range(len(image_path_list)):
                num_images_list.append(int(image_path_list[i].split(' ')[-1].split('.')[0]))  # this is specific for the structure the images are named!
            CT_patients = sorted(num_images_list)
            idx_CT_patients = [number - 1 for number in CT_patients]
            image_path_list = sorted(image_path_list, key=lambda im: int(im.split(' ')[-1].split('.')[0]))

            ### END IMAGES ###

            # read csv data
            meta_data = pd.read_csv(meta_path, delimiter=';', header=0)  # .head(180)
            overview_data = pd.read_csv(overview_path, delimiter=';', header=0)  # .head(180)
            feature_data = pd.read_csv(feat_path, delimiter=',', header=0)  # .head(180)  # loads clinical features only
            label_data = pd.read_csv(label_path, delimiter=';', header=0)  # .head(180)

            # use only patients with CT scans so far
            meta_data = meta_data.iloc[idx_CT_patients]
            overview_data = overview_data.iloc[idx_CT_patients]
            feature_data = feature_data.iloc[idx_CT_patients]
            label_data = label_data.iloc[idx_CT_patients]

            # Pre-process meta information
            meta_data = meta_data.drop(columns=['ID'])

            meta_data.Gender = meta_data.Gender.replace('Male', 0)
            meta_data.Gender = meta_data.Gender.replace('Female', 1)

            meta_data.Hospital = meta_data.Hospital.replace('Union', 0)
            meta_data.Hospital = meta_data.Hospital.replace('Liyuan', 1)

            # Pre-process feature information
            feature_data = feature_data.drop(columns=['ID'])

            # Delete features/patients with too many missing values
            # make sure to adapt yaml file accordingly!
            feature_nan_threshold = 0.8  # 0.7
            patient_nan_threshold = 1
            drop_features = []
            drop_patients = []

            for col in feature_data.columns:
                if feature_data['{}'.format(col)].isnull().sum() / len(feature_data) > feature_nan_threshold:
                    drop_features.append(col)

            feature_data = feature_data.drop(drop_features, axis=1)

            if patient_nan_threshold < 1:
                for patient in feature_data.index:
                    if feature_data.iloc[patient].isnull().sum() / len(feature_data.columns) > patient_nan_threshold:
                        drop_patients.append(patient)
          
            feature_data = feature_data.drop(drop_patients, axis=0)
            label_data = label_data.drop(drop_patients, axis=0)
            meta_data = meta_data.drop(drop_patients, axis=0)
            overview_data = overview_data.drop(drop_patients, axis=0)

            # might be necessary for filtering later
            # noCT_patients = overview_data['CT'][overview_data['CT'] != 'Positive'].index

            print()
            print('Dropped {} features from dataset [too many missing values].'.format(drop_features))
            print('Dropped {} patients from dataset [too many missing values].'.format(len(drop_patients)))

            # Pre-process labels
            label_data = label_data.drop(columns=['ID'])

            label_data["SARS-CoV-2_nucleic_acids"] = label_data["SARS-CoV-2_nucleic_acids"].replace('Positive', 1)
            label_data["SARS-CoV-2_nucleic_acids"] = label_data["SARS-CoV-2_nucleic_acids"].replace('Negative', 0)
            label_data["SARS-CoV-2_nucleic_acids"] = label_data["SARS-CoV-2_nucleic_acids"].replace(
                'Negative; Positive (Confirmed later)',
                1)  # a lot of missing data in there, which is good for evaluating MGMC's classification performance

            label_data["Mortality_outcome"] = label_data["Mortality_outcome"].replace('Deceased', 1)
            label_data["Mortality_outcome"] = label_data["Mortality_outcome"].replace('Cured', 0)
            label_data["Mortality_outcome"] = label_data["Mortality_outcome"].replace('Unknown', np.nan)

            label_data["Morbiditiy_outcome"] = label_data["Morbiditiy_outcome"].replace('Mild', 0)
            label_data["Morbiditiy_outcome"] = label_data["Morbiditiy_outcome"].replace('Regular', 1)
            label_data["Morbiditiy_outcome"] = label_data["Morbiditiy_outcome"].replace('Suspected', 2)
            label_data["Morbiditiy_outcome"] = label_data["Morbiditiy_outcome"].replace(
                'Suspected (COVID-19-confirmed later)', 3)
            label_data["Morbiditiy_outcome"] = label_data["Morbiditiy_outcome"].replace('Control', 4)
            label_data["Morbiditiy_outcome"] = label_data["Morbiditiy_outcome"].replace(
                'Control (Community-acquired pneumonia)', 5)
            label_data["Morbiditiy_outcome"] = label_data["Morbiditiy_outcome"].replace('Severe', 6)
            label_data["Morbiditiy_outcome"] = label_data["Morbiditiy_outcome"].replace('Critically ill', 7)

            # decide on which label to use for classification problem!
            # we decided to use SARS-CoV-2_nucleic_acids as classification task
            curr_label_data = label_data["SARS-CoV-2_nucleic_acids"]
            curr_label_data = curr_label_data.squeeze()

            if SCALE_METHOD == "MinMax":
                min_max_scaler = preprocessing.MinMaxScaler()
                feature_data = min_max_scaler.fit_transform(feature_data)

            elif SCALE_METHOD == "Standardization":
                feat_scaler = preprocessing.StandardScaler().fit(feature_data)
                feature_data = feat_scaler.transform(feature_data)

            meta_data['Age'] = meta_data['Age'].apply(lambda x: x / 100)

            meta_data = np.array(meta_data).astype(np.float32)
            curr_label_data = np.array(curr_label_data).astype(np.float32)
            feature_data = np.array(feature_data).astype(np.float32)

            print('Feature shape is {}'.format(feature_data.shape))
            print('Class label shape is {}'.format(curr_label_data.shape))
            print('Meta information shape is {}'.format(meta_data.shape))
            print('image_path shape is {}'.format(len(image_path_list)))

            np.save('../data/covid/img_paths.npy', image_path_list)
            np.save('../data/covid/meta_data.npy', meta_data)
            np.save('../data/covid/targets.npy', curr_label_data)
            np.save('../data/covid/features.npy', feature_data)

        self.dirs = {'img': [image_dir],
                     'img_paths': [general_dir + 'img_paths.npy'],
                     'feat': [general_dir + 'features.npy'],
                     'targets': [general_dir + 'targets.npy'],
                     'meta': general_dir + 'meta_data.npy',
                     # 'loss': [torch.nn.CrossEntropyLoss],
                     'adj': general_dir,
                     'adjtoinput': {'img': [0],
                                    'feat': [0],
                                    'seq': None}
                     }


class HAM10K(DatasetSetupBuilder):
    """#TODO"""

    def __init__(self):
        super(HAM10K, self).__init__()

    def build_dataset(self):
        general_dir = "../data/ham10k/"
        image_dir = "../data/ham10k/images/"

        print('Initializing Ham10k dataset...')

        if False:  # os.path.isdir(general_dir): #DEBUG
            print('Ham10k folder structure already created, proceeding ...')
        else:
            print('Creating Ham10k folder structure ...')
            self.create_dir('../data/')
            self.create_dir(general_dir)


            # paths to input data
            meta_path = "../data/ham10k/HAM10K_FINAL"
            meta_data =pd.read_csv(meta_path)

            # Pre-preprocess meta information
            if not os.path.exists('../data/ham10k/meta_data.npy'):
            	image_path_list=list(meta_data['path'])
            	l=[]
            	for i in range(len(image_path_list)):
            		l.append(np.asarray(Image.open(image_path_list[i]).resize((125,100))))
            	meta_images=np.array(l)
            	meta_images=meta_images.reshape(-1,125*100*3)
            	np.save('../data/ham10k/meta_data.npy', meta_images)



            # Pre-preprocess feature information
            meta_data.sex = meta_data.sex.replace('male', 0)
            meta_data.sex = meta_data.sex.replace('female', 1)
            meta_data.sex = meta_data.sex.replace('unknown',1)

            feature_data=meta_data[['age','sex']]

            #Pre-process label information
            lesion_type_dict = {
            		'nv' : 0,
            		'mel': 1,
            		'bkl': 2,
            		'bcc': 3,
            		'akiec': 4,
            		'vasc': 5,
            		'df' : 6 
            }
            d=[]
            for i in range(len(meta_data)):
            	d.append(lesion_type_dict[meta_data.loc[i,"dx"]])

            curr_label_data = pd.DataFrame(d,columns=['dx'])

            


            image_path_list = list(meta_data['path'])
            curr_label_data = np.array(curr_label_data).astype(np.float32)
            feature_data = np.array(feature_data).astype(np.float32)
          

            np.save('../data/ham10k/img_paths.npy', image_path_list)

            #np.save('../data/ham10k/meta_data.npy', meta_images)

            np.save('../data/ham10k/targets.npy', curr_label_data)
            np.save('../data/ham10k/features.npy', feature_data)

        self.dirs = {'img': [image_dir],
                     'img_paths': [general_dir + 'img_paths.npy'],
                     'feat': [general_dir + 'features.npy'],
                     'targets': [general_dir + 'targets.npy'],
                     'meta': general_dir + 'meta_data.npy',
                     # 'loss': [torch.nn.CrossEntropyLoss],
                     'adj': general_dir,
                     'adjtoinput': {'img': [0],
                                    'feat': [0],
                                    'seq': None}
                     }
