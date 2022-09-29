import argparse
import os
import sys
from glob import glob
from importlib import import_module

from omegaconf import OmegaConf


class Configuration:
    """
    # TODO
    """
    global_dict = {
        'dataset': {},
        'model': {},
        'losses': {},
        'transforms': {},
        'train': {},
        'infer': {}
    }

    def __init__(self):
        self.keyname = None
        self.args = None
        self.dirs = None
        self.parser = None
        self.get_all_configs()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        Config.global_dict['config_class'][name] = cls

    def get_all_configs(self):
        """#TODO"""
        # Get configs from argument parser
        known_args, unknown_args = self.get_argparse_args()
        known_args = OmegaConf.create(known_args.__dict__)
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        unknown_args = self._remove_extra_string(unknown_args)

        # Get configs from YAML
        yaml_conf = self.get_omegaconf_config(known_args.yaml)
        self.args = OmegaConf.merge(yaml_conf, known_args, unknown_args)

    def get_argparse_args(self):
        """#TODO"""
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--yaml', type=str, default='../base/configs/default.yaml',
                                 help='Default configuration file.')
        known_args, unknown_args = self.parser.parse_known_args()
        return known_args, unknown_args

    def include_import(self):
        module_list = self.args.include

        if not isinstance(module_list, list):
            module_list = [module_list]

        for module in module_list:
            cur_path = os.path.join(module, '*.py')
            cur_py_list = glob(cur_path, recursive=True)
            for i in cur_py_list:
                if os.path.dirname(i) not in sys.path:
                    sys.path.append(os.path.dirname(i))

                module_name = i.split(os.sep)[-1].replace('.py', '')
                import_module(module_name)

    @staticmethod
    def get_omegaconf_config(yaml: str):
        """#TODO"""
        if os.path.exists(yaml):
            conf = OmegaConf.load(yaml)
        else:
            raise FileNotFoundError('%s' % yaml)
        return conf

    @staticmethod
    def _remove_extra_string(args):
        """#TODO"""
        # Remove '--' in unknown args
        new_dict = {}
        for keys, v in args.items():
            keys = keys.replace('-', '')
            new_dict[keys] = v
        return new_dict

    @classmethod
    def to_dataset_config(cls):
        def wrapper(input_obj):
            name = input_obj.__name__
            Config.global_dict['dataset'][name] = input_obj
            return input_obj
        return wrapper


# Create Config instance here so that it is callable globally.
Config = Configuration()
