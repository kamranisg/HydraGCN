from torch import nn

from base.utils.configuration import Config


class ActivationBuilder(nn.Module):
    def __init__(self):
        super(ActivationBuilder, self).__init__()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = str(cls.__name__)
        Config.global_dict['model'][name] = cls

    def forward(self, x):
        x = self.activation(x)
        return x


class ReLU(ActivationBuilder):
    def __init__(self, *args, **kwargs):
        super(ReLU, self).__init__()
        if 'name' in kwargs:
            kwargs.pop('name')
        self.activation = nn.ReLU(*args, **kwargs)


class LeakyReLU(ActivationBuilder):
    def __init__(self, *args, **kwargs):
        super(LeakyReLU, self).__init__()
        if 'name' in kwargs:
            kwargs.pop('name')
        self.activation = nn.LeakyReLU(*args, **kwargs)


class Sigmoid(ActivationBuilder):
    def __init__(self, *args, **kwargs):
        super(Sigmoid, self).__init__()
        if 'name' in kwargs:
            kwargs.pop('name')
        self.activation = nn.Sigmoid()


class Softmax(ActivationBuilder):
    def __init__(self, *args, **kwargs):
        super(Softmax, self).__init__()
        if 'name' in kwargs:
            kwargs.pop('name')
        self.activation = nn.Softmax(*args, **kwargs, dim=0)


class Identity(ActivationBuilder):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
        if 'name' in kwargs:
            kwargs.pop('name')
        self.activation = nn.Identity(*args, **kwargs)
