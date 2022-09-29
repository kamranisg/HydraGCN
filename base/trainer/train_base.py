import copy
import os
import shutil
import time
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn.functional as F

from base.datasets.dataset_base import CustomDataLoader
from base.utils.configuration import Config
from base.observers.tensorboard import Callbacks

class TrainerBuilder:
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        Config.global_dict['train'][name] = cls

    @staticmethod
    def get_class(config):
        """#TODO"""
        if config.args.train.name in Config.global_dict.get('train'):
            return Config.global_dict.get('train').get(config.args.train.name, None)
        else:
            raise KeyError('%s is not a known key.' % config.args.train.name)


class KFoldModelTrainer(TrainerBuilder):
    def __init__(self, dataset, model, config):
        super(KFoldModelTrainer, self).__init__()
        self.phases = ['train', 'val', 'test']
        self.fold = None
        self.dataset = dataset
        self.dataset_sizes = None
        self.config = config
        self.args = config.args
        self.dataloaders = None
        self.model = model
        self.init_state_dict = copy.deepcopy(self.model.state_dict())
        self.optimizer_class = torch.optim.Adam
        self.optimizer = None
        self.loss_list = self.dataset.dirs['loss']
        print("----------------KFOLD LOSS LIST -----------------")
        print(self.loss_list)
        self.build_loss()
        self.fold_output_list = []
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.callback = Callbacks(args=config.args) if config.args.train.tensorboard else None
        if self.args.cuda:
            self.device = torch.device("cuda:" + str(self.args.GPU_device))
        else:
            self.device = torch.device("cpu")
        print('Device:', self.device)

    def reset_model(self):
        self.model.load_state_dict(self.init_state_dict)

    def run_kfold_training(self):
        fold_metric = []
        preds = []
        labels = []
        indices = []

        for f in range(self.args.folds):
            test_metrics, test_preds, test_labels, test_indices = self.run_fold(f)
            # print(test_preds)
            # print(test_labels)
            fold_metric.append(test_metrics)
            preds.extend(test_preds)
            labels.extend(test_labels)
            indices.extend(test_indices)

            # Mainly to reset callback batch and epoch counters
            self.callback.on_fold_end()

        print()
        print('+++++++++++++++++++++++++++++++++++')
        print('Model:', self.args.model_name, ', Threshold:', self.args.MNISTthreshold)
        for f in range(self.args.folds):
            print('Test Fold:', f, ', Acc:', fold_metric[f][0]['acc'], ', F1:', fold_metric[f][0]['f1'])
        print('+++++++++++++++++++++++++++++++++++')
        print()

        # preds = np.array(preds)[indices]
        # labels = np.array(labels)[indices]

        # self.test_metrics(preds, labels)

        # Save output list for plotting of feature vector input results in jupyter notebook
        self.save_outputs(self.fold_output_list)

    # def test_metrics(self, preds, labels):
    #     if self.args.train_mode == 'classification':
    #         acc = np.sum(preds == labels) / (1.0 * np.size(labels))
    #         print()
    #         print('--------------------------------')
    #         print('Test metrics:')
    #         print('Accuracy:', acc)
    #         f1score = f1_score(labels, preds, average='macro')
    #         print('F1 Score:', f1score)
    #         sensitivity = np.sum(preds[preds == labels]) / np.sum(labels)
    #         print('Sensitivity:', sensitivity)
    #         specificity = (np.sum(preds == labels) - np.sum(preds[preds == labels])) / (
    #                 np.size(labels) - np.sum(labels))
    #         print('Specificity:', specificity)
    #         print('---------------------------')
    #         print()
    #         return acc, f1score, sensitivity, specificity
    #     elif self.args.train_mode == 'regression':
    #         mae = np.sum(np.abs(preds - labels)) / (1.0 * np.size(labels))
    #         rmse = np.sqrt(np.sum(np.square(preds - labels)) / (1.0 * np.size(labels)))
    #         print()
    #         print('--------------------------------')
    #         print('Test metrics:')
    #         print('MAE:', mae)
    #         print('RMSE:', rmse)
    #         print('--------------------------------')

    def run_fold(self, fold):
        self.fold = fold
        self.dataset_sizes = {phase: self.dataset.dataset_size(fold, phase) for phase in self.phases}
        self.dataloaders = {p: CustomDataLoader(self.dataset,
                                                args=self.args,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False,
                                                fold=fold,
                                                phase=p) for p in self.phases}

        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)

        self.reset_model()  # Essential so that the fold is always started with an untrained model
        self.model.to(self.device)

        # Train model
        t_total = time.time()
        best_checkpoint = 0.0
        best_loss = 100000.0
        best_state_dict = {}

        # Initially impute and normalize feature vector input once using the train dataloader for full batch training
        if self.args.dataset.preprocess_feat:
            key_list = ['train', 'val', 'test'] if self.args.batchtype == 'batch' else ['train']
            for k, v in enumerate(key_list):
                cur_dataloader = self.dataloaders[v]
                if v == 'train':
                    cur_dataloader.feat_preprocess()
                else:
                    preprocessor_cls_list = self.dataloaders['train'].preprocessor_cls_list
                    cur_dataloader.feat_preprocess(preprocessor_cls_list=preprocessor_cls_list)

        patience_c = 0
        extracted_features = np.zeros((len(self.dataset), self.args.extract_dim))
        extracted_features_list = {}
        indices_list = {}
        for epoch in range(self.args.epochs):
            print("------------------EPOCHS--------------")
            print(self.args.epochs)
            print(self.args)
            since = time.time()
            print()
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Fold {}/{}, epoch {}/{} (Patience: {}, currently {})'.format(self.fold + 1, self.args.folds,
                                                                                epoch + 1, self.args.epochs,
                                                                                self.args.patience, patience_c))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                epoch_loss, outputs, labels, indices_list[phase], extracted_features_list[phase] = self.run_epoch(phase, fold)
                print("-------------------outputs----------------")
                print(outputs)
                epoch_metrics = self.metrics(epoch_loss, outputs, labels, phase, self.config)

            # print('+++++++++++++++++++++++++++++++')
            # print(sorted(indices_list['train']))
            # print(sorted(indices_list['val']))
            # print('+++++++++++++++++++++++++++++++')

            time_elapsed = time.time() - since
            print('Epoch complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()

            if epoch_metrics[0]['checkpointer'] > best_checkpoint or epoch_loss < best_loss or not best_checkpoint:
                if epoch_metrics[0]['checkpointer'] > best_checkpoint or not best_checkpoint:
                    best_checkpoint = epoch_metrics[0]['checkpointer']
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                patience_c = 0
                best_state_dict = copy.deepcopy(self.model.state_dict())

                if len(extracted_features_list['val']) > 0:
                    extracted_features[indices_list['train']] = np.array(extracted_features_list['train'])
                    extracted_features[indices_list['val']] = np.array(extracted_features_list['val'])

            else:
                patience_c += 1

            # self.checkpointer(epoch, fold, epoch_metrics, best_loss, freq=100)
            if self.callback is not None:
                self.callback.on_epoch_end()

            if patience_c > self.args.patience:
                print('Training is not improving performance anymore, training is stopped.')
                break

        print("Optimization Finished!")
        total_time = time.time() - t_total
        print("Total time elapsed: {:.0f}m {:.0f}s".format(
            total_time // 60, total_time % 60))
        print("Running test set prediction ...")

        self.model.load_state_dict(best_state_dict)
        test_loss, preds, labels, indices_list['test'], extracted_features_list['test'] = self.run_epoch('test', fold)

        if len(extracted_features_list['test']) > 0:
            extracted_features[indices_list['test']] = np.array(extracted_features_list['test'])
            # print(extracted_features)
            # print(extracted_features.shape)
            # print(bla)
            np.save(self.args.extract_file.format(self.args.dataset.name, fold, self.args.batch_size,
                                                  self.args.label_list), extracted_features)

        preds_list = self.calculate_pred(preds)

        # Save only classification output for now
        self.fold_output_list.append((torch.tensor(labels[0]).squeeze(), preds_list[0], indices_list['test']))
        test_metrics = self.metrics(test_loss, preds, labels, 'test', self.config)
        return test_metrics, preds_list, labels, indices_list['test']

    def run_epoch(self, phase, fold):
        """
        This function runs one full epoch of the training for one phase using the function self.epoch_step

        :param phase: the phase of the current training step, either train, val or test
        :return: returns the accumulated loss, the predictions and the corresponding labels
        of the full epoch in two torch tensors
        """
        running_loss = 0.0
        outputs_list = []
        labels_list = []
        indices_list = []
        extracted_features_list = []

        self.dataloaders[phase].set_phase()

        for step, input_tuple in enumerate(self.dataloaders[phase]):
            # print(input_tuple)
            loss_value, curr_outputs, curr_labels, curr_indices = self.epoch_step(phase, input_tuple, fold)
            running_loss += loss_value
            internal_indices = curr_outputs[0]['indices']  # since it is the same for all
            indices_list.extend(curr_indices[internal_indices].tolist())
            for i in range(len(curr_outputs)):
                if step == 0:
                    outputs_list.append([])
                    labels_list.append([])
                outputs_list[i].extend(curr_outputs[i]['input'][internal_indices].tolist())
                labels_list[i].extend(curr_labels[i][internal_indices].tolist())

                for key in curr_outputs[i].keys():
                    if key == 'extracted_features':
                        extracted_features_list.extend(curr_outputs[i]['extracted_features'][internal_indices].tolist())  # contains only correct entries belonging to phase
                        # print('++++++++++ length ++++++++++++ :', len(extracted_features_list), len(extracted_features_list[step*250]))
                        # print(np.array(extracted_features_list)[step*250:20+step*250])
                        # print('++++++++++ length ind ++++++++++++ :', len(internal_indices_list))

            # if type(curr_labels) is list and len(curr_labels) == 1:
            #     curr_labels = curr_labels[0]
            # else:
            #     # Remove this if we just want to run feat or img input for now. self.metric currently only supports
            #     # single-target output.
            #     raise NotImplementedError('From this point onwards multi-label is not yet supported.')
            # labels.extend(curr_labels.tolist())

        epoch_loss = running_loss / self.dataset_sizes[phase]

        return epoch_loss, outputs_list, labels_list, indices_list, extracted_features_list

    def epoch_step(self, phase, input_tuple, fold):
        batch_input_list, target_list, indices = self.dataloaders[phase].prepare_batch(input_tuple)
        # Load tensors into device

        if 'Extract' in self.args.dataset.name:
            fold_batch_input = [batch_input_list[fold]]
            for m in range(self.args.folds, len(batch_input_list)):
                fold_batch_input.append(batch_input_list[m])
            batch_input_list = fold_batch_input

        for idx, inp in enumerate(batch_input_list):
            for k, v in inp.items():
                # Get edge idx and edge weights of adjacency matrix.
                if k == 'adj' and v is not None:
                    v = self.dataset.get_edges_of_batch_idx(v, indices, phase)  # TODO currently no val - val connection
                elif k == 'adj_wts' and v is not None:
                    v = self.dataset.get_edges_weights_of_batch_idx(v, indices, phase)
                inp[k] = v.to(self.device)
            batch_input_list[idx] = inp

        for k, v in enumerate(target_list):
            target_list[k] = v.to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # training with dropout and back propagation only in phase 'train'
        if phase == 'train':
            # batch_input_copy = copy.deepcopy(batch_input_list)
            self.model.train()  # Set model to training mode
            with torch.set_grad_enabled(True):
                outputs = self.model(batch_input_list)
                for i, output in enumerate(outputs):
                    s_rate = self.loss_list[i].get('supervision_rate', 1.0) # what is this?
                    if Config.args.batchtype in ['full', 'full_train']:
                        curr_indices = torch.tensor(self.dataset.index_folds[self.fold][phase]).long()
                        curr_indices = curr_indices[:int(curr_indices.size(0)*s_rate)]
                        #     num_semi = int(output['input'].size(0) * Config.args.supervision_rate)
                        #     output['indices'] = indices[:num_semi]
                    elif Config.args.batchtype in ['batch', 'batch_train']:
                        curr_indices = torch.tensor(np.arange(output['input'].size(0))).long()
                        curr_indices = self.dataset.get_supervision(s_rate, indices, curr_indices, fold)
                        # output['input'] = output['input'][indices]
                        # output['nan_idx'] = output['nan_idx'][indices]
                        # target_list[i] = target_list[i][indices]
                    else:
                        raise ValueError('Batch type unknown')
                    output['indices'] = curr_indices
                if len(outputs[0]['input'].size()) == 3:
                    outputs[0]['input'] = outputs[0]['input'].squeeze()
                loss = self.calc_loss(outputs, target_list, phase)
                # backward + optimize only if in training phase
                loss.backward()
                # debug
                # for name, param in self.model.named_parameters():
                #     print(name, torch.isfinite(param.grad).all())
                # end debug
                self.optimizer.step()

                # Log training outputs
                if self.callback is not None:
                    self.callback.on_batch_end(phase, self.fold, outputs, target_list, self.loss_list)

        # calculate the outcome for train and val without dropout
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_input_list)
            for i, output in enumerate(outputs):
                if Config.args.batchtype in ['full', 'full_train']:
                    curr_indices = torch.tensor(self.dataset.index_folds[self.fold][phase]).long()
                elif Config.args.batchtype in ['batch', 'batch_train']:
                    length = output['input'].size(0)
                    if not phase == 'train':
                        length -= self.args.batch_size // 2
                    curr_indices = torch.tensor(np.arange(length)).long()
                    # if phase in ['val', 'test']:
                    #     output['nan_idx'] = output['nan_idx'][indices]
                    #     target_list[i] = target_list[i][indices]
                else:
                    raise ValueError('Batch type unknown')
                output['indices'] = curr_indices
            if len(outputs[0]['input'].size()) == 3:
                outputs[0]['input'] = outputs[0]['input'].squeeze()

            print_loss = self.calc_loss(outputs, target_list, 'eval')
        # Log validation/test outputs
        if self.callback is not None:
            self.callback.on_batch_end(phase, self.fold, outputs, target_list, self.loss_list)

        input_size = outputs[0]['indices'].size(0)
        running_loss = print_loss.item() * input_size

        return running_loss, outputs, target_list, indices

    def calc_loss(self, outputs, labels, phase):
        loss_value = 0.0
        print("--------------------CALC LOSS ----------------")
        print(self.loss_list)
        for i, output in enumerate(outputs):
            # loss_wrapper = self.get_loss_wrapper(self.config.train.mode[i])
            # loss_value += loss_wrapper(self.loss_list[i], outputs[i], labels[i])
            # {'name': 'CrossEntropyLoss', 'wts': 1.0}
            # if Config.args.batchtype == 'batch':

            # Simulate semi-supervision during training and in full batch mode
            # if phase == 'train':  # and Config.args.batchtype == 'full':
            #     supervision_rate = self.loss_list[i].get('supervision_rate', 1.0)
            #     num_semi = int(len(output['indices']) * supervision_rate)
            #     output['indices'] = output['indices'][:num_semi]  # TODO This works only for full, since order of samples is changing for batch
            
            print(self.loss_list[i]['loss'])
            print("output")

            loss_value += self.loss_list[i]['loss'](output, labels[i]) * self.loss_list[i]['wts']
            # elif Config.args.batchtype == 'full':
            #     indices = self.dataset.index_folds[self.fold][phase]
            #     indices = torch.tensor(indices)
            #     # output = {k: v[indices] for k, v in output.items()}
            #     # output['input'] = output['input'][indices]
            #     # output.update({'indices': indices})
            #     loss_value += self.loss_list[i]['loss'](output, labels[i][indices]) * self.loss_list[i]['wts']
            # else:
            #     raise ValueError('Incorrect batpe')
        return loss_value

    def calculate_pred(self, preds):
        preds_list = []
        for i, loss_ in enumerate(self.loss_list):
            cur_pred = preds[i]
            mode = loss_['loss'].mode
            predictor_cls = Config.global_dict['train']['{}Pred'.format(mode)]
            predictor = predictor_cls()
            preds_list.append(predictor(cur_pred))
        return preds_list

    def metrics(self, epoch_loss, outputs, labels, phase, config):
        metrics_list = []
        for i, loss_ in enumerate(self.loss_list):
            cur_output = outputs[i]
            cur_label = labels[i]
            mode = loss_['loss'].mode
            print("-------------MODE IN METRIC----------------")
            print(mode)

            if 'metric' in Config.args.dirs and Config.args.dirs.metric[i] is not None:
                metric_cls = Config.global_dict['train'][Config.args.dirs.metric[i]['metric']]
                print(metric_cls)
            else:
                metric_cls = Config.global_dict['train']['{}Metrics'.format(mode)]
            metric = metric_cls()
            metrics_list.append(metric(epoch_loss, cur_output, cur_label, phase, config, i))
        return metrics_list

    def checkpointer(self, epoch, fold, epoch_metrics, best_loss, freq=100,
                     fpath=None):
        """Save last epoch and best model if validation metric improves."""
        if epoch % freq == 0:
            cur_val_ = epoch_metrics['checkpointer']
            is_best = bool(cur_val_ < best_loss)
            state = {'epoch': epoch,
                     'state_dict': self.model.state_dict(),
                     'optim_dict': self.optimizer.state_dict()}
            if fpath is None:
                fpath = './outputs/%s/checkpoint/' % self.args.dataset
            filepath = os.path.join(fpath, 'last_%s.pth.tar' % fold)
            if not os.path.exists(fpath):
                print('Checkpoint Directory does not exist! Making directory %s'
                      % fpath)
                os.makedirs(fpath)
            else:
                print("Checkpoint Directory exists! ")
            torch.save(state, filepath)
            if is_best:
                print("Validation metric improved. Saving new best model!")
                best_fname = 'best_%s.pth.tar' % fold
                shutil.copyfile(filepath,os.path.join(fpath, best_fname))

    def save_outputs(self, output):
        """Save output at output_dir."""
        output_dir = './outputs/{}/test_pred/{}SimulatedMissing_{}SupervisionRate/'.format(self.args.dataset.name,
                                                                                           self.args.p_missing,
                                                                                           self.args.supervision_rate)
        output_fpath = os.path.join(output_dir, 'kfold_output_{}.pkl'.format(self.config.args.model_name))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_fpath, 'wb') as fpath:
            print('Saving outputs...')
            torch.save(output, fpath)
            if os.path.exists(output_fpath):
                print('=== SAVED %s ===' % output_fpath)

    def build_loss(self):
        """#TODO"""
        self.loss_list = list(self.loss_list)
        self.loss_list = [dict(x) for x in self.loss_list]
        for idx, loss_ in enumerate(self.loss_list):
            loss_fn = Config.global_dict['losses'][loss_['loss']]
            self.loss_list[idx]['loss'] = loss_fn(loss_)
