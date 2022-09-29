import os
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import pairwise_distances_chunked
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from torchvision import datasets, transforms
from torch.utils.data import sampler, DataLoader

from base.utils.configuration import Config
#from base.utils import utils as dut  # Added by Ekin
#from scipy import ndimage # Added by Alex
import omegaconf
from omegaconf import OmegaConf


class DatasetBuilder(datasets.ImageFolder):
    def __init__(self, config):
        self.config = config
        self.args = config.args
        self.phase = 'train'
        self.dirs = config.args.dirs
        self.targets_list = []
        self.img_paths_list = []
        self.features_list = []
        self.sequences_list = []
        self.meta_data = None
        self.miss_features_list = []  # List of feature matrices with simulated feature level missingness
        self.sim_feat_nan_idx_list = []  # List of mask matrix for known entries which were simulated missing
        self.feat_obs_idx_list = []  # List of mask matrix for known entries which are originally observed
        self.filter_map = None
        self.adj_dim = None
        self._load_data()
        self.num_data = self.targets_list[0].size(0)
        if self.dirs['img']:
            self.transforms = self.get_transforms()
        else:
            self.transforms = None
        # self.target_shape = []
        self.input_shape_list = []
        self.target_shape_list = []

        self.index_folds = self.create_indices()
        self.init_all_adjs = self.load_adj()
        self.get_input_shape()  # Store input shapes to config
        self.get_target_shape()  # Store target shapes to config

        # Added by Ekin
        #self.mask_dir = "/home/ubuntu/local-s3-bucket/lung_masks/"

    @staticmethod
    def get_class(config):
        """#TODO"""
        if config.args.dataset.datatype in Config.global_dict.get('dataset'):
            return Config.global_dict.get('dataset').get(config.args.dataset.datatype, None)
        else:
            raise KeyError('%s is not a known key.' % config.args.dataset.datatype)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        Config.global_dict['dataset'][name] = cls

    def get_input_shape(self):
        print("--------SELF--------")
        for i in self[0][0]:
            #print("i" +str(i))
            inp = i['input']
            inp_shape = tuple(inp.unsqueeze(0).shape)
            print("INSIDE DATA BASE "  +str(inp_shape))
            self.input_shape_list.append(inp_shape)
        Config.args['input_shape_list'] = self.input_shape_list
        Config.args['len_input'] = len(self)

    @staticmethod
    def get_loss_mode():
        mode_list = []
        loss_list = Config.args.dirs['loss']
        for k, loss in enumerate(loss_list):
            loss_class = Config.global_dict['losses'][loss['loss']]
            mode = loss_class({'dummy': None}).mode
            mode_list.append(mode)
        return mode_list

    def get_mfb(self):
        mfb_list = []
        mode_list = self.get_loss_mode()
        for k, mode in enumerate(mode_list):
            if mode == 'Classification':
                # TODO implement median frequency balancing
                mfb_list.append(None)
            else:
                mfb_list.append(None)
        Config.args['mfb_weights'] = mfb_list

    def get_target_shape(self):
        mode_list = self.get_loss_mode()
        for k, mode in enumerate(mode_list):
            if mode == 'Classification':
                cur_target_size = torch.unique(self.targets_list[k]).size(0)
                if cur_target_size == 2:
                    self.target_shape_list.append(1)  # 1  # why is it like that?
                elif cur_target_size > 2:
                    self.target_shape_list.append(cur_target_size)
                else:
                    raise ValueError('Incorrect target size.')
            elif mode == 'Regression':
                target_shape = self.targets_list[k].size(1)
                self.target_shape_list.append(target_shape)
            elif mode == 'Segmentation':
                raise NotImplementedError
            else:
                raise ValueError('Incorrect mode')
        Config.args['target_shape_list'] = self.target_shape_list

    def _filter_labels(self):
        path = self.dirs['targets'][0]
        cur_target = np.load(path)
        self.adj_dim = cur_target.shape[0]
        if self.args.label_list:
            self.filter_map = np.zeros(cur_target.shape[0]).astype(bool)
            for label in self.args.label_list:
                self.filter_map[cur_target == label] = True
        else:
            self.filter_map = np.ones(cur_target.shape[0]).astype(bool)

    def _create_extraction_folder(self):
        dir_split = self.dirs['targets'][0].split('/')
        dir_ = '{}/{}/{}Extract/'.format(dir_split[0], dir_split[1], dir_split[2])
        if not os.path.isdir(dir_):
            print('Create directory', dir_)
            os.mkdir(dir_)
        return dir_

    def _load_data(self):

        # Collect all image paths given list of image dirs

        self._filter_labels()
        # if 'Extract' in Config.args.curr_model_name: # comment out since only used in loop_experiments.py
        #    extract_dir = self._create_extraction_folder() # comment out since only used in loop_experiments.py

        if self.dirs['img']:
            # if len(self.dirs['img']) >= 2:
            #     raise NotImplementedError('Multi input img not yet implemented and tested. Check todo below')
            print("--------------dir-------------------")
            print(self.dirs['img'])
            super(DatasetBuilder, self).__init__(self.dirs['img'][0])  # only necessary to get self.loader
            for i in range(len(self.dirs['img'])):
                # TODO check if there is an issue when init in for loop.
                # super(DatasetBuilder, self).__init__(i)
                ## img_paths = [s[0] for s in self.samples]
                #print(i,len(self.dirs['img']))
                #print(self.dirs['img_paths'][i])
                img_paths = list(np.load(self.dirs['img_paths'][i],allow_pickle=True))  # [self.filter_map])  # TODO label_list needed
                self.img_paths_list.append(img_paths)

        # Load given list of feature matrix
        if self.dirs['feat']:
            for i in self.dirs['feat']:
                self.features_list.append(
                    torch.tensor(np.load(i)[self.filter_map], dtype=torch.float32))  # TODO label_list needed
                # if 'Extract' in Config.args.curr_model_name:
                #    np.save(extract_dir + 'features_l{}.npy'.format(self.args.label_list), np.load(i)[self.filter_map])
                # Simulate missingness if p_missing in default config is between (0.0, 1.0)
                # Will randomly drop p_missing percentage of features per sample/row.
                if self.args.dataset.get('p_missing', None) is not None and (self.args.dataset.p_missing > 0.0) and (
                        self.args.dataset.p_missing < 1.0):
                    self.simulate_missingness()
                    self.set_feat_observe_mask_list()
                else:
                    # List of feature matrix mask specifying which entries are observed.
                    self.set_feat_observe_mask_list()

        # Load given list of sequences
        if self.dirs['seq']:
            for i in self.dirs['seq']:
                self.sequences_list.append(torch.tensor(np.load(i)[self.filter_map], dtype=torch.float32))  # TODO label_list needed

        # Combine targets from img dataset and given target list
        if self.dirs['targets']:
            for path in self.dirs['targets']:
                cur_target = torch.tensor(
                    np.load(path, allow_pickle=True))  # [self.filter_map])  # TODO label_list needed !! added dtype
                if self.args.label_list:
                    for new_label, label in enumerate(self.args.label_list):
                        cur_target[cur_target == label] = new_label
                self.targets_list.append(cur_target)
                # if 'Extract' in Config.args.curr_model_name:
                #    np.save(extract_dir + 'targets_l{}.npy'.format(self.args.label_list), np.load(path)[self.filter_map])

        self.meta_data = torch.tensor(np.load(self.dirs['meta'])[self.filter_map])  # TODO label_list needed
        # if 'Extract' in Config.args.curr_model_name:
        #    np.save(extract_dir + 'meta_data_l{}.npy'.format(self.args.label_list), np.load(self.dirs['meta'])[self.filter_map])

    def create_indices(self):
        test_per = 0.15
        index_folds = []

        if self.args.folds > 1:
            idx = np.arange(self.num_data)
            if self.args.train_mode == 'classification':
                skf = StratifiedKFold(n_splits=self.args.folds, shuffle=True, random_state=0)
            else:
                skf = KFold(n_splits=self.args.folds, shuffle=True, random_state=0)
            for train_index, test_index in skf.split(idx, self.targets_list[0].numpy()):
                if self.args.train_mode == 'classification':
                    stratifier = self.targets_list[0].numpy()[train_index]
                else:
                    stratifier = None
                train_id, val_id, _, _ = train_test_split(np.array(train_index), np.array(train_index),
                                                          stratify=stratifier,
                                                          test_size=test_per, random_state=0)
                test_id = test_index
                indices = {'train': train_id,
                           'val': val_id,
                           'test': test_id
                           }
                index_folds.append(indices)
        else:
            idx = np.arange(self.num_data)
            if self.args.train_mode == 'classification':
                stratifier = self.targets_list[0].numpy()
            else:
                stratifier = None
            train_index, test_id, _, _ = train_test_split(np.array(idx), np.array(self.targets_list[0].numpy()),
                                                          stratify=stratifier,
                                                          test_size=test_per, random_state=0)
            if self.args.train_mode == 'classification':
                stratifier = self.targets_list[0].numpy()[train_index]
            else:
                stratifier = None
            train_id, val_id, _, _ = train_test_split(np.array(train_index), np.array(train_index),
                                                      stratify=stratifier,
                                                      test_size=test_per, random_state=0)
            indices = {'train': train_id,
                       'val': val_id,
                       'test': test_id
                       }
            index_folds.append(indices)
        return index_folds

    def get_sampler(self, fold, phase):
        used_sampler = sampler.SubsetRandomSampler(self.index_folds[fold][phase])
        return used_sampler

    def dataset_size(self, fold, phase):
        return np.size(self.index_folds[fold][phase])

    # def set_supervision_rate(self):
    #     targets = self.targets_list[0].numpy()
    #     supervision_indices = []
    #     for ind_dict in self.index_folds:
    #         curr_targets = targets[ind_dict['train']]
    #         num_classes = self.target_shape_list[0]
    #         ind_range = np.arange(curr_targets.shape[0])
    #         s_indices = []
    #         for label in range(num_classes):
    #             curr_ind = ind_range[curr_targets == label]
    #             np.random.shuffle(curr_ind)
    #             s_indices.extend(curr_ind)
    #         supervision_indices.append(np.array(s_indices))
    #     return supervision_indices

    def get_supervision(self, s_rate, indices, internal_indices, fold):
        size = self.index_folds[fold]['train'].size
        mask = np.in1d(indices.numpy(), self.index_folds[fold]['train'][:int(size * s_rate)])
        if not np.any(mask):
            print('++++++++++++++++++++++++++++++++++++++')
            print('---------------------------- No supervision found for this batch!!! -------------------------------')
            print('++++++++++++++++++++++++++++++++++++++')
            mask[:2] = True
        s_indices = internal_indices[mask]
        return s_indices

    def load_adj(self):
        print('Columns of meta data used for graph creation:', self.args.meta_columns)
        all_adjs = []
        for meta_col in self.args.meta_columns:
            adj_dir = self.get_adj_file(meta_col)
            if os.path.exists(adj_dir):
                print('Adjacency already has been created.')
                print('Loading adjacency ...')
                edges = np.load(adj_dir)
                edge_wts = np.load(adj_dir.replace('.npy', 'wts.npy'))
                print("DATABASE LOAD ADJ")
                print(edges.shape)
                print(edge_wts.shape)
                print('Adjacency loaded.')
            else:
                print('Creating adjacency for meta data column', meta_col, ' ...')
                edges, edge_wts = self.create_adj(meta_col, adj_dir)

            print('Creating affinity matrix ...')
            adj = sp.coo_matrix((edge_wts, (edges[:, 0], edges[:, 1])),
                                shape=(self.adj_dim, self.adj_dim),  # TODO Dimension problem under label list
                                dtype=np.float32)
            print('Transferring matrix to dense representation ...')
            print(adj.shape)
            adj = np.array(adj.todense())  # TODO label_list needed
            print(adj.shape)
            adj = adj[self.filter_map, :]
            adj = adj[:, self.filter_map]
            # if 'Extract' in Config.args.curr_model_name:
            #    self.save_extracted_adj(adj, meta_col)
            # self.edge_statistics(adj, meta_col)
            all_adjs.append(torch.tensor(adj))

        all_adjs = torch.stack(all_adjs)
        if not self.args.multigraph:
            all_adjs = torch.sum(all_adjs, 0)
            # all_adjs = all_adjs == len(self.args.meta_columns)  # Only if all conditions are fulfilled the edge remains
            # all_adjs = torch.unsqueeze(all_adjs, 0).int()
            all_adjs = torch.unsqueeze(all_adjs, 0).float()
        # self.edge_statistics(all_adjs[0, :, :].numpy(), 'all')
        return all_adjs

    def get_adj_file(self, meta_col):
        if 'Extract' not in Config.args.dataset.name:
            adj_dir = self.dirs['adj'] + '/adj.npy'
            adj_dir = adj_dir.replace('.npy', '_' + str(meta_col) + '.npy')
        else:
            adj_dir = self.dirs['adj']
            adj_dir = adj_dir.replace('adj', 'adj_' + str(meta_col))
        return adj_dir

    def save_extracted_adj(self, adj, meta_col):
        dir_split = self.dirs['targets'][0].split('/')
        dir_ = '{}/{}/{}Extract/'.format(dir_split[0], dir_split[1], dir_split[2])
        edges = adj.nonzero()
        edge_weights = adj[edges]
        edges = np.stack(edges).T
        edges = np.array(edges)
        if len(edges) == 0:
            print('No edge found!')
            edges = [[0, 0]]
        extract_dir = dir_ + 'adj_{}_l{}.npy'.format(meta_col, self.args.label_list)
        np.save(extract_dir, edges)
        np.save(extract_dir.replace('.npy', 'wts.npy'), edge_weights)

    def create_adj(self, meta_col, adj_dir):
        edge_criterion_class = EdgeCriterionBuilder.get_class(self.config)
        adj = edge_criterion_class.edge_criterion(self.meta_data, meta_col=meta_col)
        adj = adj.astype('float32')
        edges = adj.nonzero()
        edge_weights = adj[edges]
        edges = np.stack(edges).T
        edges = np.array(edges)
        if len(edges) == 0:
            print('No edge found!')
            edges = [[0, 0]]
        np.save(adj_dir, edges)
        np.save(adj_dir.replace('.npy', 'wts.npy'), edge_weights)
        print('Adjacency created.')
        return edges, edge_weights

    def edge_statistics(self, adj, meta_col):
        if len(self.targets_list) > 1 or len(list(self.targets_list[0].shape)) > 1:
            raise NotImplementedError  # only single target single class implemented at the moment
        else:
            true_dist = np.abs(self.targets_list[0].numpy() - self.targets_list[0].numpy()[:, None])
            adj_56 = np.abs(self.targets_list[0].numpy() + self.targets_list[0].numpy()[:, None]) == 3
            adj_36 = np.abs(self.targets_list[0].numpy() - self.targets_list[0].numpy()[:, None]) == 2
            adj_35 = np.abs(self.targets_list[0].numpy() + self.targets_list[0].numpy()[:, None]) == 1
            true_adj = true_dist == 0
            corr_adj = true_adj.astype(np.int32) - adj.astype(np.int32)
            inter_56 = adj_56.astype(np.int32) - adj.astype(np.int32)
            inter_36 = adj_36.astype(np.int32) - adj.astype(np.int32)
            inter_35 = adj_35.astype(np.int32) - adj.astype(np.int32)

        correct_edges = np.sum((corr_adj + true_adj.astype(np.int32)) == 1) - np.size(true_adj, 0)  # subtract self-connections
        false_edges = -1 * np.sum(corr_adj[corr_adj == -1])
        missed_edges = np.sum(corr_adj[corr_adj == 1])
        connected = np.sum((np.sum(adj.astype(np.int32) - np.eye(self.num_data), 1) > 0).astype(np.int32))
        unconnected = np.sum((np.sum(adj.astype(np.int32) - np.eye(self.num_data), 1) == 0).astype(np.int32))
        inter56_edges = np.sum((inter_56 + adj_56.astype(np.int32)) == 1)
        inter36_edges = np.sum((inter_36 + adj_36.astype(np.int32)) == 1)
        inter35_edges = np.sum((inter_35 + adj_35.astype(np.int32)) == 1)

        print()
        print('+++++++++++++++++++++++')
        print('Edge statistics for meta column {}'.format(meta_col))
        print('Possible edges:', self.num_data * self.num_data)
        print('Correct edges:', correct_edges)
        print('False edges:', false_edges)
        print('Missed edges:', missed_edges)
        print('Edges between 5 and 6 (only 3,5,6 MNIST):', inter56_edges)
        print('Edges between 3 and 6 (only 3,5,6 MNIST):', inter36_edges)
        print('Edges between 3 and 5 (only 3,5,6 MNIST):', inter35_edges)
        print('Connected nodes:', connected)
        print('Unconnected nodes:', unconnected)
        print('+++++++++++++++++++++++')
        print()

    def get_transforms(self):
        yaml_file = self.config.args.model.yaml_path
        if os.path.exists(yaml_file):
            model_config = OmegaConf.load(yaml_file)
            model_config = model_config[list(model_config.keys())[0]]
            model_name = model_config.DynamicBlock0.ParallelCNN.Model0.model
            transforms_class = Config.global_dict['transforms']['{}Transforms'.format(model_name)]
            return transforms_class()
        else:
            raise ValueError('Model yaml file not found')

    def __getitem__(self, index):
        # model_name = Config.args.train.model
        # self.transforms_class = Config.global_dict['transforms']['{}Transforms'.format(model_name)]
        # self.transforms = transforms_class().get_transforms
        inputs_list = []

        # Get full adjacency vector at current index.
        adj = self.init_all_adjs[:, index, :]

        # Get image at current index in given list of image paths
        if self.img_paths_list is not None and len(self.img_paths_list) > 0:
            for i, img_path in enumerate(self.img_paths_list):


                img = self.loader(img_path[index]) ## load images

                #name = img_path[index].split('cohort1')[-1]
                #mask_dir = "/home/ubuntu/local-s3-bucket/lung_masks/"  # make generic later
                #mask = np.load(mask_dir+name)
                #roi_limits = dut.extract_lung_area(mask)
                #img = np.load(img_path[index])  # self.loader(img_path[index])
                #img = img[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]

                #img = img.astype(np.float32) / 255


                #### TRANSFORMATION USING ZOOM ###
                #nshape = (128, 128, 64)
                #desired_depth = nshape[2]
                #desired_width = nshape[1]
                #desired_height = nshape[0]

                # Get current depth
                #current_depth = img.shape[-1]
                #current_width = img.shape[0]
                #current_height = img.shape[1]

                # Compute depth factor
                #depth = current_depth / desired_depth
                #width = current_width / desired_width
                #height = current_height / desired_height
                #if depth == 0:
                #    print("Zero")
                #depth_factor = 1 / depth
                #width_factor = 1 / width
                #height_factor = 1 / height

                # Resize across z-axis
                #img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)

                #### END TRANSFORMATION USING ZOOM ####

                #img = torch.from_numpy(img)

                #print(img.size) # for Pillow images

                if self.transforms:
                    self.transforms.set_phase(self.phase)
                    img = self.transforms(img)
                adj_idx = self.dirs['adjtoinput']['img'][i]
                curr_adj = adj[adj_idx, :].unsqueeze(0)  # TODO specify which adj belongs to with input, edge features

                # Within forward method of a model, 'adj' will contain edge idx, 'adj_wts' will contain edge weights.
                inputs_list.append({'input': img, 'adj': curr_adj, 'adj_wts': curr_adj})

        # Get feature vector at current index in given list of features
        if self.features_list is not None and len(self.features_list) > 0:
            for i, feat in enumerate(self.features_list):

                adj_idx = self.dirs['adjtoinput']['feat'][i]
                curr_adj = adj[adj_idx, :].unsqueeze(0)  # TODO specify which adj belongs to with input, edge features

                # Simulate feature-wise missingness if p within (0, 1)
                if self.args.dataset.get('p_missing', None) is not None and (self.args.dataset.p_missing > 0.0) and (
                        self.args.dataset.p_missing < 1.0):

                    inputs_list.append({'input': self.miss_features_list[i][index], 'adj': curr_adj,
                                        'adj_wts': curr_adj, 'orig_input': feat[index],
                                        'sim_feat_nan_mask': self.sim_feat_nan_idx_list[i][index]})
                else:
                    inputs_list.append({'input': feat[index], 'adj': curr_adj, 'adj_wts': curr_adj,
                                        'feat_observe_mask': self.feat_obs_idx_list[i][index]})

        # Get sequence at current index in given list of sequence
        if self.sequences_list is not None and len(self.sequences_list) > 0:
            for i, seq in enumerate(self.sequences_list):
                adj_idx = self.dirs['adjtoinput']['seq'][i]
                curr_adj = adj[adj_idx, :].unsqueeze(0)  # TODO specify which adj belongs to with input, edge features

                # Within forward method of a model, 'adj' will contain edge idx, 'adj_wts' will contain edge weights.
                inputs_list.append({'input': seq[index], 'adj': curr_adj, 'adj_wts': curr_adj})

        # Get targets at current index given list of targets.
        target_list = []
        for k, v in enumerate(self.targets_list):
            # TODO in-multi output we might need to specify which target is for classification/regression.
            if self.args.train_mode == 'classification':
                target_list.append(v[index].long())
            else:
                target_list.append(v[index].float())

        # inputs_list = [{'input': img, 'adj': adj}, {'input': img, 'adj': adj}, {'input': feat, 'adj': adj}]
        return inputs_list, target_list, index

    def __len__(self):
        return self.targets_list[0].shape[0]

    def get_edges_of_batch_idx(self, adj, batch_idx, phase):
        """Given adjacency matrix, remove nodes and edges which are not part of the batch."""
        curr_adjs = torch.clone(adj)
        # Take nodes of interest and remove the connections between validation
        # and test samples
        if curr_adjs.shape[1] == 1:
            curr_adjs = curr_adjs[:, 0, :]
            curr_adjs = curr_adjs[:, batch_idx.long()]
            if not phase == 'train' and self.args.batchtype in ['batch', 'batch_train']:
                curr_adjs[:batch_idx.size(0) - self.args.batch_size // 2, :] = 0  # TODO removes val connections
            curr_adjs = curr_adjs.nonzero().t().contiguous()
        else:
            # TODO
            raise NotImplementedError('Edge Feature vector adjacency not yet supported')
        return curr_adjs

    def get_edges_weights_of_batch_idx(self, adj, batch_idx, phase):
        """Given adjacency matrix, remove nodes and edges which are not part of the batch and return edge weights"""
        curr_adjs = torch.clone(adj)
        # Take nodes of interest and remove the connections between validation
        # and test samples
        if curr_adjs.shape[1] == 1:
            curr_adjs = curr_adjs[:, 0, :]
            curr_adjs = curr_adjs[:, batch_idx.long()]
            if not phase == 'train' and self.args.batchtype in ['batch', 'batch_train']:
                curr_adjs[:batch_idx.size(0) - self.args.batch_size // 2, :] = 0  # TODO removes val connections
            idx = curr_adjs.nonzero()
            edge_weights = curr_adjs[idx[:, 0], idx[:, 1]]
        else:
            # TODO
            raise NotImplementedError('Edge Feature vector adjacency not yet supported')
        return edge_weights

    def simulate_missingness(self):
        """Simulate missingness for tabular data"""
        for k, feat in enumerate(self.features_list):

            # Get known_blocks matrix denoting which blocks are currently known
            known_mask = ~torch.isnan(feat)
            num_blocks = len(Config.args.dataset.feat_block_list[k].keys())
            known_blocks = torch.zeros(feat.shape[0], num_blocks)
            for key_, val_ in enumerate(Config.args.dataset.feat_block_list[k].items()):
                block_not_missing = known_mask[:, val_[1]].all(1).reshape(-1, 1)
                known_blocks[:, key_:key_ + 1] = block_not_missing.float()

            # Get mask_blocks matrix denoting which entries should remain
            blocks_prob = torch.ones_like(known_blocks) - Config.args.dataset.p_missing
            blocks_prob = blocks_prob * known_blocks
            cur_block_mask = torch.bernoulli(blocks_prob)
            mask_blocks = torch.zeros_like(feat)
            for key_, val_ in enumerate(Config.args.dataset.feat_block_list[k].items()):
                mask_blocks[:, val_[1]] = cur_block_mask[:, key_].reshape(-1, 1)

            assert bool(((known_mask.float() - mask_blocks) == -1).any()) is False, 'mask_blocks is incorrect'
            miss_feat = torch.where(mask_blocks.bool(), feat, torch.tensor(float('nan')))
            self.miss_features_list.append(miss_feat)
            self.sim_feat_nan_idx_list.append(known_mask.float() - mask_blocks)

    def set_feat_observe_mask_list(self):
        for k, feat in enumerate(self.features_list):
            self.feat_obs_idx_list.append(~torch.isnan(feat))

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if hasattr(self, 'root') and self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)


class CustomDataLoader:  # (DataLoader)
    def __init__(self, dataset, args, shuffle, num_workers, drop_last, fold, phase):

        self.args = args
        self.phase = phase
        self.feat_num_idx_list = None
        self.feat_cat_idx_list = None
        self.preprocessor_cls_list = []
 
        if self.args.batchtype in ['full', 'full_train']:
            self.sampler = None
            length = len(dataset)
        elif self.args.batchtype in ['batch', 'batch_train']:
            self.sampler = dataset.get_sampler(fold, phase)
            length = 64 #self.args.batch_size
            if not self.phase == 'train':
                print("----------------LENGTH----------------")
                print(length)
                length = length // 2
                self.train_dataloader = DataLoader(dataset, batch_size=length, shuffle=shuffle,
                                                   num_workers=num_workers, drop_last=drop_last,
                                                   sampler=dataset.get_sampler(fold, 'train'))
        else:
            raise ValueError('Batch type {} unknown'.format(self.args.batchtype))

        self.dataloader = DataLoader(dataset, batch_size=length, shuffle=shuffle,
                                     num_workers=num_workers, drop_last=drop_last,
                                     sampler=self.sampler)

    def __iter__(self):
        return self.dataloader.__iter__()

    def prepare_batch(self, input_tuple):
        if self.phase == 'train' or self.args.batchtype in ['full', 'full_train']:
            return input_tuple
        else:
            # input_tuple = next(iter(self.dataloader.__iter__()))
            batch_input_list, target_list, indices = input_tuple

            train_input_tuple = next(iter(self.train_dataloader.__iter__()))
            train_batch_input_list, train_target_list, train_indices = train_input_tuple

            new_batch_input_list = []
            new_target_list = []

            for i, d in enumerate(batch_input_list):
                new_batch_input_list.append({})
                for k, v in d.items():
                    new_batch_input_list[i][k] = torch.cat((v, train_batch_input_list[i][k]), dim=0)
            for i, v in enumerate(target_list):
                new_target_list.append(torch.cat((v, train_target_list[i]), dim=0))
            new_indices = torch.cat((indices, train_indices), dim=0)

            return new_batch_input_list, new_target_list, new_indices

    def set_phase(self):
        print('Entering phase', self.phase, '...')
        # self.dataset.phase = self.phase
        self.dataloader.dataset.phase = self.phase

    def get_dataset(self):
        return self.dataloader.dataset

    def set_feat_num_cat_idx_list(self):
        """This will set a list which defines which are numerical/categorical variables in every feature vector list."""

        if Config.args.get('feat_num_idx_list', None) is None:
            self.feat_num_idx_list = [np.arange(x.shape[-1]) for x in self.dataloader.dataset.features_list]
        else:
            self.feat_num_idx_list = Config.args.dataset.get('feat_num_idx_list')

        if Config.args.get('feat_cat_idx_list', None) is not None:
            self.feat_cat_idx_list = Config.args.dataset.get('feat_cat_idx_list')
        else:
            self.feat_cat_idx_list = [None for _ in range(len(self.dataloader.dataset.features_list))]

    def feat_preprocess(self, preprocessor_cls_list=None):
        """Mean imputation and zero-mean unit variance scaling pre-processing step."""
        self.set_feat_num_cat_idx_list()
        p_missing = Config.args.dataset.p_missing
        for k, v in enumerate(self.dataloader.dataset.features_list):
            num_idx_ = self.feat_num_idx_list[k]
            cat_idx_ = self.feat_cat_idx_list[k]
            if self.phase == 'train':
                preprocessor_cls = self.get_feat_preprocessor_cls(num_feat_idx=num_idx_, cat_feat_idx=cat_idx_)

                # If missingness is simulated use parameters of input features with missingness for preprocessing
                # else just use original input features' parameters
                if (p_missing is not None) and (p_missing > 0.0) and (p_missing < 1.0):
                    if self.dataloader.dataset.args.batchtype == 'batch':
                        preprocessor_cls.fit(self.dataloader.dataset.miss_features_list[k][self.sampler.indices])
                    else:
                        preprocessor_cls.fit(self.dataloader.dataset.miss_features_list[k])
                    cur_data_ = self.dataloader.dataset.miss_features_list[k].numpy()
                    self.dataloader.dataset.miss_features_list[k] = torch.tensor(preprocessor_cls.transform(cur_data_)).float()
                else:
                    if self.dataloader.dataset.args.batchtype == 'batch':
                        preprocessor_cls.fit(v[self.sampler.indices])
                    else:
                        preprocessor_cls.fit(v.numpy())

                self.dataloader.dataset.features_list[k] = torch.tensor(preprocessor_cls.transform(v.numpy())).float()
                self.preprocessor_cls_list.append(preprocessor_cls)

            else:
                if (p_missing is not None) and (p_missing > 0.0) and (p_missing < 1.0):
                    self.dataloader.dataset.miss_features_list[k] = preprocessor_cls_list[k].fit_transform(
                        self.dataloader.dataset.miss_features_list[k])
                    # Addition for new loader:
                    if not self.phase == 'train':
                        self.train_dataloader.dataset.miss_features_list[k] = preprocessor_cls_list[k].fit_transform(
                            # TODO check if this works!!!
                            self.train_dataloader.dataset.miss_features_list[k])
                self.dataloader.dataset.features_list[k] = torch.tensor(
                    preprocessor_cls_list[k].fit_transform(v.numpy())).float()
                # Addition for new loader:
                self.train_dataloader.dataset.features_list[k] = torch.tensor(preprocessor_cls_list[k].fit_transform(
                    self.train_dataloader.dataset.features_list[k].numpy())).float()  # TODO check if this works!!!

    @staticmethod
    def get_feat_preprocessor_cls(num_feat_idx=None, cat_feat_idx=None):
        """Pre-processing pipeline.

        Will give a sklearn scaler object which can perform mean imputation and zero-mean unit variance scaling.
        """
        # Set remainder to passthrough to use all other columns or drop to remove columns not in feat_idx and cat_idx.
        remainder = Config.args.get('feat_remainder', 'drop')

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', preprocessing.StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='missing')),
            ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore'))])
        transformers = list()
        if num_feat_idx is not None:
            transformers.append(('num', numeric_transformer, num_feat_idx))
        if cat_feat_idx is not None:
            transformers.append(('cat', categorical_transformer, cat_feat_idx))
        preprocessor = ColumnTransformer(transformers=transformers, remainder=remainder)
        return preprocessor


class EdgeCriterionBuilder:

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        Config.global_dict['dataset'][name] = cls

    @staticmethod
    def edge_criterion(meta_data, meta_col):
        raise NotImplementedError

    @staticmethod
    def get_class(config):
        """#TODO"""
        if config.args.dataset.edge_criterion in Config.global_dict.get('dataset'):
            return Config.global_dict.get('dataset').get(config.args.dataset.edge_criterion, None)
        else:
            raise KeyError('%s is not a known key.' % config.args.dataset.edge_criterion)


class MNISTEdgeCriterion(EdgeCriterionBuilder):
    def __init__(self):
        super(MNISTEdgeCriterion, self).__init__()

    @staticmethod
    def edge_criterion(meta_data, meta_col):
        threshold = [Config.args.MNISTthreshold]  # 750, 1000
        meta_data = meta_data.type(torch.float32)
        pairwise_iter = pairwise_distances_chunked(meta_data, metric='l2', n_jobs=-2)
        for i, it in enumerate(pairwise_iter):
            if i == 0:
                dist = it
            else:
                dist = np.concatenate((dist, it), 0)
        edges = dist <= threshold[meta_col]
        return edges


class ToysetEdgeCriterion(EdgeCriterionBuilder):
    def __init__(self):
        super(ToysetEdgeCriterion, self).__init__()

    @staticmethod
    def edge_criterion(meta_data, meta_col):
        threshold = [2, 0, 5]  # age, gender, weight
        dist = np.abs(meta_data[:, meta_col] - meta_data[:, meta_col, None])
        edges = dist <= threshold[meta_col]
        return edges.numpy()


class TADPOLEEdgeCriterion(EdgeCriterionBuilder):
    def __init__(self):
        super(TADPOLEEdgeCriterion, self).__init__()

    @staticmethod
    def edge_criterion(meta_data, meta_col):
        threshold = [5, 0, 0]  # 'AGE', 'PTGENDER', 'APOE4' columns
        dist = np.abs(meta_data[:, meta_col] - meta_data[:, meta_col, None])
        edges = dist <= threshold[meta_col]
        return edges.numpy()


class TADPOLEMeanImputedEdgeCriterion(EdgeCriterionBuilder):
    def __init__(self):
        super(TADPOLEMeanImputedEdgeCriterion, self).__init__()

    @staticmethod
    def edge_criterion(meta_data, meta_col):
        threshold = [0, 2, 0, 0, 0]  # 'SITE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4'
        dist = np.abs(meta_data[:, meta_col] - meta_data[:, meta_col, None])
        edges = dist <= threshold[meta_col]
        return edges.numpy()


class TADPOLESplittedEdgeCriterion(EdgeCriterionBuilder):
    def __init__(self):
        super(TADPOLESplittedEdgeCriterion, self).__init__()

    @staticmethod
    def edge_criterion(meta_data, meta_col):
        threshold = [2, 0, 0]  # 'AGE', 'PTGENDER', 'APOE4'
        dist = np.abs(meta_data[:, meta_col] - meta_data[:, meta_col, None])
        edges = dist <= threshold[meta_col]
        return edges.numpy()


class ChestXray14SLEdgeCriterion(EdgeCriterionBuilder):
    def __init__(self):
        super(ChestXray14SLEdgeCriterion, self).__init__()

    @staticmethod
    def edge_criterion(meta_data, meta_col):
        threshold = [0, 5, 0, 0]  # PatID, Age, Gender, Position
        dist = np.abs(meta_data[:, meta_col] - meta_data[:, meta_col, None])
        edges = dist <= threshold[meta_col]
        return edges.numpy()


class UCIThyroidEdgeCriterion(EdgeCriterionBuilder):
    def __init__(self):
        super(UCIThyroidEdgeCriterion, self).__init__()

    @staticmethod
    def edge_criterion(meta_data, meta_col):
        """

        Notes:
            'age',                              # continuous in range (0, 1]
            'sex',                              # binary 0/1
            'on thyroxine',                     # binary 0/1
            'query on thyroxine',               # binary 0/1
            'on antithyroid medication',        # binary 0/1
            'sick',                             # binary 0/1
            'pregnant',                         # binary 0/1
            'thyroid surgery',                  # binary 0/1
            'I131 treatment',                   # binary 0/1
            'query hypothyroid',                # binary 0/1
            'query hyperthyroid',               # binary 0/1
            'lithium',                          # binary 0/1
            'goitre',                           # binary 0/1
            'tumor',                            # binary 0/1
            'hypopituitary',                    # binary 0/1
            'psych',                            # binary 0/1
        """
        threshold = [0] * 16
        dist = np.abs(meta_data[:, meta_col] - meta_data[:, meta_col, None])
        edges = dist <= threshold[meta_col]
        return edges.numpy()


class COVIDiCTCFEdgeCriterion(EdgeCriterionBuilder):
    def __init__(self):
        super(COVIDEdgeCriterion, self).__init__()

    @staticmethod
    def edge_criterion(meta_data, meta_col):
        threshold = [0, 0.05, 0]  # 'HOSPITAL', 'AGE', 'GENDER' columns
        dist = np.abs(meta_data[:, meta_col] - meta_data[:, meta_col, None])
        edges = dist <= threshold[meta_col]
        return edges.numpy()

class HAM10KEdgeCriterion(EdgeCriterionBuilder):
    def __init__(self):
        super(HAM10KEdgeCriterion,self).__init__()
        
    @staticmethod
    def edge_criterion(meta_data, meta_col):
        threshold = [0, 5, 0]  # 'HOSPITAL', 'AGE', 'GENDER' columns
        dist = np.abs(meta_data[:, meta_col] - meta_data[:, meta_col, None])
        edges = dist <= threshold[meta_col]
        return edges.numpy()


# class NHANESEdgeCriterion(EdgeCriterionBuilder):
#     def __init__(self):
#         super(NHANESEdgeCriterion, self).__init__()
#
#     @staticmethod
#     def edge_criterion(meta_data, meta_col):
#         """
#

#         Notes:
#             (https://github.com/mkachuee/Opportunistic/blob/04edd60cdfc56a6490bfa7634f7bb2282c6d6035/nhanes.py#L401)
#             # 0 			real - gender (was normalized)
#             # 1 	 		real - age
#             # Additional information which may be relevant to diabetes
#             # 2-7 			binary
#             # 8-12			binary
#             # 12-16		    binary
#         """
#         threshold = [0] * 17
#         raise NotImplementedError('Pairwise similarity metric OOM')
#         # # dist = np.abs(meta_data[:, meta_col] - meta_data[:, meta_col, None])
#         # edges = dist <= threshold[meta_col]
#         # return edges.numpy()
