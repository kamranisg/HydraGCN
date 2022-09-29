import sys
sys.path.append('../')
import itertools

from omegaconf import OmegaConf

from base.utils.configuration import Config
from base.datasets.dataset_setup import DatasetSetupBuilder
from base.datasets.dataset_base import DatasetBuilder
from base.models.models import ModelBuilder
from base.trainer.train_base import TrainerBuilder
import copy


def main():
    # Will include user-defined classes in gloabl_dict
    Config.include_import()
    config_model_name = copy.deepcopy(Config.args.model.name)
    # Update configuration according to desired experiments.
    dataset_list = ['TADPOLESplitted']
    model_list = ['HydraGNN']
    model_yaml_dir = '../base/configs/{}/{}.yaml'

    supervision_rate_list = [.05, .1, .25, .5]
    # supervision_rate_list = [1.0]

    all_exp_list = list(itertools.product(dataset_list, model_list, supervision_rate_list))
    for exp_i in all_exp_list:
        Config.args.model.name = config_model_name
        dataset_ = exp_i[0]
        model_ = exp_i[1]
        supervision_rate = exp_i[2]

        # Update configuration in default yaml
        update_default_yaml(Config, dataset_, model_, model_yaml_dir, supervision_rate)

        # Setup dataset folder
        builder_class = DatasetSetupBuilder.get_class(Config)
        builder_class().setup_data()

        # Update dirs in dataset setup
        update_dirs(Config, dataset_, model_)

        # Full or batch-wise dataset
        dataset_class = DatasetBuilder.get_class(Config)
        dataset = dataset_class(Config)

        # Create model architecture
        model_class = ModelBuilder.get_class(Config)
        model = model_class(Config)

        # Training
        trainer_class = TrainerBuilder.get_class(Config)
        trainer = trainer_class(dataset, model, Config)
        trainer.run_kfold_training()


# Helper functions to modify configuration
def update_default_yaml(conf, dataset_name, model_name, model_yaml_dir, supervision_rate):

    # Supervision rate is used during SSL and when saving SSL results
    conf.args.supervision_rate = supervision_rate
    conf.args.folds = 10
    conf.args.patience = 30
    conf.args.seed = 0
    conf.args.curr_model_name = model_name
    conf.args.train.n_jobs = 1

    if dataset_name == 'TADPOLESplitted':
        # Change default.yaml config
        conf.args.dataset.name = dataset_name
        conf.args.dataset.edge_criterion = dataset_name + 'EdgeCriterion'
        conf.args.batchtype = 'full'
        conf.args.multigraph = False
        conf.args.meta_columns = [0, 1, 2]
        conf.args.model.yaml_path = model_yaml_dir.format(dataset_name, model_name)
        conf.args.dataset.preprocess_feat = True

        if model_name in ['MultiGCN']:
            conf.args.multigraph = True

    else:
        raise ValueError('dataset_name unknown {}'.format(dataset_name))

    # NHANES
    # Thyroid
    # ...


def update_dirs(conf, dataset_name, model_name):

    # Mean imputed and standardized to zero-mean and unit variance scaling
    if dataset_name == 'TADPOLESplitted':

        if model_name in ['PyGCN', 'PyCDGM']:
            conf.args.dirs.adjtoinput.feat = [0, 0, 0, 0]
            conf.args.dirs.targets = [conf.args.dirs.targets[0]]
            conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1.0}]

        elif model_name in ['HydraGNN']:
            conf.args.dirs.adjtoinput.feat = [0, 0, 0, 0]

            # Dummy targets
            conf.args.dirs.targets = [conf.args.dirs.targets[0],
                                      conf.args.dirs.feat[0],
                                      conf.args.dirs.feat[1],
                                      conf.args.dirs.feat[2],
                                      conf.args.dirs.feat[3]]
            conf.args.dirs.loss = [
                {'loss': "CrossEntropyLoss", 'wts': 1.0, 'supervision_rate': conf.args.supervision_rate},
                {'loss': "MSELoss", 'wts': .2, 'keyname': 'input', 'target_keyname':
                    'target_output'},
                {'loss': "MSELoss", 'wts': .2, 'keyname': 'input', 'target_keyname':
                    'target_output'},
                {'loss': "MSELoss", 'wts': .2, 'keyname': 'input', 'target_keyname':
                    'target_output'},
                {'loss': "MSELoss", 'wts': .2, 'keyname': 'input', 'target_keyname':
                    'target_output'},
            ]

            conf.args.dirs.metric = [
                {'metric': "MultiTaskClassifierMetric"},
                None,
                None,
                None,
                None
            ]

        elif model_name in ['HydraGNNClassifier']:
            conf.args.dirs.adjtoinput.feat = [0, 0, 0, 0]

            # Dummy targets
            conf.args.dirs.targets = [
                conf.args.dirs.targets[0],
                conf.args.dirs.feat[0],
                conf.args.dirs.feat[1],
                conf.args.dirs.feat[2],
                conf.args.dirs.feat[3]
            ]
            conf.args.dirs.loss = [
                {'loss': "CrossEntropyLoss", 'wts': 1.0, 'supervision_rate': conf.args.supervision_rate},
                {'loss': "MSELoss", 'wts': .0, 'keyname': 'input', 'target_keyname':
                    'target_output'},
                {'loss': "MSELoss", 'wts': .0, 'keyname': 'input', 'target_keyname':
                    'target_output'},
                {'loss': "MSELoss", 'wts': .0, 'keyname': 'input', 'target_keyname':
                    'target_output'},
                {'loss': "MSELoss", 'wts': .0, 'keyname': 'input', 'target_keyname':
                    'target_output'},
            ]
            conf.args.dirs.metric = [{'metric': "ClassificationMetrics"},
                                     None,
                                     None,
                                     None,
                                     None]

        else:
            raise NotImplementedError('Using {} for {} not implemented'.format(model_name, dataset_name))

    else:
        raise ValueError('{} dataset unknown'.format(dataset_name))
    # NHANES
    # Thyroid
    # ...
if __name__ == "__main__":
    main()
