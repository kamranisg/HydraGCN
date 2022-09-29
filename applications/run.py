import sys
sys.path.append('../')
from base.utils.configuration import Config
from base.datasets.dataset_setup import DatasetSetupBuilder
from base.datasets.dataset_base import DatasetBuilder
from base.models.models import ModelBuilder
from base.trainer.train_base import TrainerBuilder

# from pydoc import locate
# import inspect


def main():
    # debug
    # dot_path = 'base.datasets.dataset_base.DatasetBuilder'
    # transform_class = locate(dot_path)
    # print(transform_class)
    # assert inspect.isclass(transform_class), "Could not load {}".format(dot_path)
    # print(stop)
    # enddebug

    print('Starting run.py ...')
    print()


    # Will include user-defined classes in gloabl_dict
    Config.include_import()

    # Setup dataset folder
    builder_class = DatasetSetupBuilder.get_class(Config)
    builder_class().setup_data()

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

    print('Done with run.py.')

if __name__ == "__main__":
    main()
