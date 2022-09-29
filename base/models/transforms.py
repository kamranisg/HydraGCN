from torchvision import transforms
import torchvision
from torch import nn
from pydoc import locate
import inspect

from base.utils.configuration import Config


class TransformsBuilder:
    def __init__(self):
        super(TransformsBuilder, self).__init__()
        self.data_transform = None
        self.data_transforms = None  # Dict of transforms for train, val, test phases

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = str(cls.__name__)
        Config.global_dict['transforms'][name] = cls

    def set_phase(self, phase):
        self.data_transform = self.data_transforms[phase]

    def __call__(self, x):
        x = self.data_transform(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PyCNNTransforms(TransformsBuilder):
    def __init__(self, *args, **kwargs):
        super(PyCNNTransforms, self).__init__()
        user_transforms = Config.args.user_transforms

        phases = ['train', 'val', 'test']
        self.data_transforms = {}

        for phase in phases:
            transforms_list = []
            if phase == 'train':
                for i in user_transforms:
                    transform = list(dict(i).keys())[0]
                    params = i[transform]
                    dot_path = 'torchvision.transforms.{}'.format(transform)
                    transform_class = locate(dot_path)
                    assert inspect.isclass(transform_class), "Could not load".format(dot_path)
                    transforms_list.append(transform_class(**params))
            # transforms_list.append(transforms.Resize((30, 30)))
            transforms_list.append(transforms.ToTensor())

            self.data_transforms[phase] = transforms.Compose(transforms_list)


class AlexNetTransforms(TransformsBuilder):
    def __init__(self, *args, **kwargs):
        super(AlexNetTransforms, self).__init__()
        user_transforms = Config.args.user_transforms

        phases = ['train', 'val', 'test']
        self.data_transforms = {}

        for phase in phases:
            transforms_list = []
            if phase == 'train':
                for i in user_transforms:
                    transform = list(dict(i).keys())[0]
                    params = i[transform]
                    dot_path = 'torchvision.transforms.{}'.format(transform)
                    transform_class = locate(dot_path)
                    assert inspect.isclass(transform_class), "Could not load".format(dot_path)
                    transforms_list.append(transform_class(**params))
            transforms_list.append(transforms.Resize((224, 224))) ## resize to mosaic size
            transforms_list.append(transforms.ToTensor())
            transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)

            self.data_transforms[phase] = transforms.Compose(transforms_list)


class ResNetTransforms(TransformsBuilder):
    def __init__(self, *args, **kwargs):
        super(ResNetTransforms, self).__init__()
        user_transforms = Config.args.user_transforms

        phases = ['train', 'val', 'test']
        self.data_transforms = {}

        for phase in phases:
            transforms_list = []
            if phase == 'train':
                for i in user_transforms:
                    transform = list(dict(i).keys())[0]
                    params = i[transform]
                    dot_path = 'torchvision.transforms.{}'.format(transform)
                    transform_class = locate(dot_path)
                    assert inspect.isclass(transform_class), "Could not load".format(dot_path)
                    transforms_list.append(transform_class(**params))
            transforms_list.append(transforms.Resize((224, 224)))
            transforms_list.append(transforms.ToTensor())

            self.data_transforms[phase] = transforms.Compose(transforms_list)


class VGGTransforms(TransformsBuilder):
    def __init__(self, *args, **kwargs):
        super(VGGTransforms, self).__init__()
        user_transforms = Config.args.user_transforms

        phases = ['train', 'val', 'test']
        self.data_transforms = {}

        for phase in phases:
            transforms_list = []
            if phase == 'train':
                for i in user_transforms:
                    transform = list(dict(i).keys())[0]
                    params = i[transform]
                    dot_path = 'torchvision.transforms.{}'.format(transform)
                    transform_class = locate(dot_path)
                    assert inspect.isclass(transform_class), "Could not load".format(dot_path)
                    transforms_list.append(transform_class(**params))
            transforms_list.append(transforms.Resize((224, 224)))
            transforms_list.append(transforms.ToTensor())

            self.data_transforms[phase] = transforms.Compose(transforms_list)


class DenseNetTransforms(TransformsBuilder):
    def __init__(self, *args, **kwargs):
        super(DenseNetTransforms, self).__init__()
        user_transforms = Config.args.user_transforms

        phases = ['train', 'val', 'test']
        self.data_transforms = {}

        for phase in phases:
            transforms_list = []
            if phase == 'train':
                for i in user_transforms:
                    transform = list(dict(i).keys())[0]
                    params = i[transform]
                    dot_path = 'torchvision.transforms.{}'.format(transform)
                    transform_class = locate(dot_path)
                    assert inspect.isclass(transform_class), "Could not load".format(dot_path)
                    transforms_list.append(transform_class(**params))
            transforms_list.append(transforms.Resize((224, 224)))
            transforms_list.append(transforms.ToTensor())

            self.data_transforms[phase] = transforms.Compose(transforms_list)
