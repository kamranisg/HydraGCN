import copy

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
# from base.models.model_base import CNN, GAT, SGAT, Skip, AfterGAT
from pydoc import locate
import inspect
import torch_geometric.nn as tg
import torch_geometric
import omegaconf

from base.utils.configuration import Config


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

    def forward(self, *args):
        raise NotImplementedError

    @staticmethod
    def get_class(config):
        if config.args.model.name in Config.global_dict.get('model'):
            return Config.global_dict.get('model').get(config.args.model.name, None)
        else:
            raise KeyError('%s is not a known key.' % config.args.model.name)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        Config.global_dict['model'][name] = cls

    @staticmethod
    def set_kwargs(conf, layer, idx):
        if idx is not None:
            kwargs = {}
            for key, val in conf['layer{}'.format(layer)].items():
                if isinstance(val, omegaconf.listconfig.ListConfig):
                    kwargs[key] = list(val)[idx]
                elif isinstance(val, int):
                    kwargs[key] = val
                else:
                    raise ValueError()
        else:
            kwargs = conf['layer{}'.format(layer)]
        return kwargs


class PyGCN(ModelBuilder):
    def __init__(self, conf, idx):
        super(PyGCN, self).__init__()
        gnn_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            gnn_list.append(tg.GCNConv(**kwargs))
        self.model_list = nn.ModuleList(gnn_list)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj, adj_wts = x_dict['input'], x_dict['adj'], x_dict.get('adj_wts', None)
        for model in self.model_list:
            x = model(x, adj, adj_wts)
            x = self.activation(x)
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyChebConv(ModelBuilder):
    def __init__(self, conf, idx):
        super(PyChebConv, self).__init__()
        gnn_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            gnn_list.append(tg.ChebConv(**kwargs))
        self.model_list = nn.ModuleList(gnn_list)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj, adj_wts = x_dict['input'], x_dict['adj'], x_dict.get('adj_wts', None)
        for model in self.model_list:
            x = model(x, adj, adj_wts)
            x = self.activation(x)
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyGAT(ModelBuilder):
    def __init__(self, conf, idx):
        super(PyGAT, self).__init__()
        gnn_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            gnn_list.append(tg.GATConv(**kwargs))
        self.model_list = nn.ModuleList(gnn_list)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj, adj_wts = x_dict['input'], x_dict['adj'], x_dict.get('adj_wts', None)

        for model in self.model_list:
            x = self.activation(x)
            x = model(x, adj)
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PySAGEConv(ModelBuilder):
    def __init__(self, conf, idx):
        super(PySAGEConv, self).__init__()
        gnn_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            gnn_list.append(tg.SAGEConv(**kwargs))
        self.model_list = nn.ModuleList(gnn_list)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj, adj_wts = x_dict['input'], x_dict['adj'], x_dict.get('adj_wts', None)
        for model in self.model_list:
            x = self.activation(x)
            x = model(x, adj, adj_wts)
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyMLP(ModelBuilder):
    def __init__(self, conf, idx):
        super(PyMLP, self).__init__()
        mlp_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            mlp_list.append(nn.Linear(**kwargs))
        self.model_list = nn.ModuleList(mlp_list)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']

        for model in self.model_list:
            x = self.activation(x)
            x = model(x)
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyMLPExtract(ModelBuilder):
    def __init__(self, conf, idx):
        super(PyMLPExtract, self).__init__()
        mlp_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            mlp_list.append(nn.Linear(**kwargs))
        self.model_list = nn.ModuleList(mlp_list)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']
        out_dict = {k: v for k, v in x_dict.items()}
        for i, model in enumerate(self.model_list):
            if i == len(self.model_list)-1:
                # print('x:', x)
                # print(x.shape)
                out_dict['extracted_features'] = x
            x = self.activation(x)
            # print('x activ:', x)
            x = model(x)
            # print('x final:', x)
            # print('extracted:', out_dict['extracted_features'])
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyPass(ModelBuilder):
    def __init__(self, conf, idx):
        super(PyPass, self).__init__()

    def forward(self, x):
        return x


class ResNet(ModelBuilder):
    def __init__(self, conf, idx):
        super(ResNet, self).__init__()
        dot_path = 'torchvision.models.resnet{}'.format(conf.resnet)
        resnet_class = locate(dot_path)
        assert inspect.isfunction(resnet_class), "Could not load {}".format(dot_path)
        kwargs = self.set_kwargs(conf, 0, idx)
        self.model = resnet_class(**kwargs)

        # Reset last fc-layer:
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=conf.out_features, bias=True)
        nn.init.xavier_uniform_(self.model.fc.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']
        x = self.model(x)
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class AlexNet(ModelBuilder):
    def __init__(self, conf, idx):
        super(AlexNet, self).__init__()
        kwargs = self.set_kwargs(conf, 0, idx)
        self.model = torchvision.models.alexnet(**kwargs)

        # Reset last fc-layer:
        self.model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, conf.out_features),
        )

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']
        x = self.model(x)
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class VGG(ModelBuilder):
    def __init__(self, conf, idx):
        super(VGG, self).__init__()
        dot_path = 'torchvision.models.vgg{}'.format(conf.vgg)
        model_cls = locate(dot_path)
        assert inspect.isfunction(model_cls), "Could not load {}".format(dot_path)
        kwargs = self.set_kwargs(conf, 0, idx)
        self.model = model_cls(**kwargs)

        # Reset last fc-layer:
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features=in_features, out_features=conf.out_features, bias=True)
        nn.init.xavier_uniform_(self.model.classifier[-1].weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']
        x = self.model(x)
        # x = {'input': x, 'adj': adj}
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class DenseNet(ModelBuilder):
    def __init__(self, conf, idx):
        super(DenseNet, self).__init__()
        dot_path = 'torchvision.models.densenet{}'.format(conf.densenet)
        model_cls = locate(dot_path)
        assert inspect.isfunction(model_cls), "Could not load {}".format(dot_path)
        kwargs = self.set_kwargs(conf, 0, idx)
        self.model = model_cls(**kwargs)

        # Reset last fc-layer:
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=in_features, out_features=conf.out_features, bias=True)
        nn.init.xavier_uniform_(self.model.classifier.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']
        x = self.model(x)
        # x = {'input': x, 'adj': adj}
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyCNN(ModelBuilder):
    def __init__(self, conf, idx):
        super(PyCNN, self).__init__()
        cnn_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            cnn_list.append(nn.Sequential(nn.Conv2d(**kwargs), nn.BatchNorm2d(kwargs.get('out_channels'))))
        self.model_list = nn.ModuleList(cnn_list)
        self.pool = nn.MaxPool2d(kernel_size=2)
        mlp_in_features = conf.mlp_layer.in_features.get('in_features', 190512) #10*12) # CHANGED!! find generic value for this
        self.fc = nn.Linear(in_features=mlp_in_features, out_features=conf.mlp_layer.out_features)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']
        x = torch.mean(x, 1)
        x = x.unsqueeze(1)
        for model in self.model_list:
            x = model(x)
            print(x.shape)
            x = self.pool(x)
            x = self.activation(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        print('shape before FC Layer starts:')
        print(x.shape)
        print()
        x = self.fc(x)
        x = self.activation(x)
        # x = {'input': x, 'adj': adj}
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyUpConv(ModelBuilder):
    def __init__(self, conf, idx):
        super(PyUpConv, self).__init__()
        # channel = 16
        # height = 5
        # width = 5
        # self.fc = nn.Linear(in_features=conf.layer0.in_features, out_features=channel*height*width)
        # self.activation = Config.global_dict['model'][conf.activation.name]
        # self.activation = self.activation(**conf.activation)
        # self.unflatten = nn.Unflatten(1, (channel, height, width))
        dcnn_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            if layer == 0:
                self.channel = kwargs.get('channels', 16)
                self.height = kwargs.get('height', 5)
                self.width = kwargs.get('width', 5)
                self.fc = nn.Linear(in_features=conf.layer0.in_features, out_features=self.channel * self.height * self.width)
                self.activation = Config.global_dict['model'][conf.activation.name]
                self.activation = self.activation(**conf.activation)
                # self.unflatten = nn.Unflatten(1, (self.channel, self.height, self.width))
            else:  # Since the first layer is the MLP layer, which is then unflattened to feature maps
                # kwargs = self.set_kwargs(conf, layer, idx)
                dcnn_list.append(nn.Sequential(nn.ConvTranspose2d(**kwargs), nn.BatchNorm2d(kwargs.get(
                    'out_channels'))))
        self.model_list = nn.ModuleList(dcnn_list)
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.unpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']
        # x = torch.mean(x, 1)
        # x = x.unsqueeze(1)
        x = self.fc(x)
        x = self.activation(x)
        x = x.reshape((-1, self.channel, self.height, self.width))
        # x = self.unflatten(x)
        for i, model in enumerate(self.model_list):
            x = model(x)
            if i == (len(self.model_list) - 1):
                x = F.sigmoid(x)
            else:
                x = self.activation(x)
            # x = self.upsample(x)
            #x = self.unpool(x)  # TODO get the unpool indices here, if UNET-type upsampling should take place
        # x = {'input': x, 'adj': adj}
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyLSTM(ModelBuilder):
    """
    # TODO
    Notes:
        1. Using layers parameter in config will build N layers of LSTM.
        2. num_layers param in LSTM will build N layers of stacked LSTM.
        2. Will always return tensor with shape (batch_dim, 1, output_feature_dim). Which is the output at the last
            time step.
        3. There is always a linear layer and activation function after RNN/LSTM.
        4. Initial hidden state and cell state input will be initialized with zeros.
    """
    def __init__(self, conf, idx):
        super(PyLSTM, self).__init__()
        rnn_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            kwargs['batch_first'] = True
            rnn_list.append(nn.LSTM(**kwargs))
        self.model_list = nn.ModuleList(rnn_list)
        fc_kwargs = conf.fc_layer
        self.fc = nn.Linear(**fc_kwargs)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']
        for model in self.model_list:
            x, _ = model(x)
        x = x[:, -1, :]  # Take the last timestep's output
        x = self.fc(x)
        x = self.activation(x)
        # x = {'input': x, 'adj': adj}
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyGRU(ModelBuilder):
    """
    # TODO
    """
    def __init__(self, conf, idx):
        super(PyGRU, self).__init__()
        rnn_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            kwargs['batch_first'] = True
            rnn_list.append(nn.GRU(**kwargs))
        self.model_list = nn.ModuleList(rnn_list)
        fc_kwargs = conf.fc_layer
        self.fc = nn.Linear(**fc_kwargs)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']
        for model in self.model_list:
            x, _ = model(x)
        x = x[:, -1, :]  # Take the last timestep's output
        x = self.fc(x)
        x = self.activation(x)
        # x = {'input': x, 'adj': adj}
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyRNN(ModelBuilder):
    """
    # TODO
    """
    def __init__(self, conf, idx):
        super(PyRNN, self).__init__()
        rnn_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            kwargs['batch_first'] = True
            rnn_list.append(nn.RNN(**kwargs))
        self.model_list = nn.ModuleList(rnn_list)
        fc_kwargs = conf.fc_layer
        self.fc = nn.Linear(**fc_kwargs)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']
        for model in self.model_list:
            x, _ = model(x)
        x = x[:, -1, :]  # Take the last timestep's output
        x = self.fc(x)
        x = self.activation(x)
        # x = {'input': x, 'adj': adj}
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyDecoderGRU(ModelBuilder):
    """
    GRU decoder for autoencoder style network architectures.

    """
    def __init__(self, conf, idx):
        super(PyDecoderGRU, self).__init__()
        rnn_list = []
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            kwargs['batch_first'] = True
            rnn_list.append(nn.GRU(**kwargs))
        self.model_list = nn.ModuleList(rnn_list)
        self.timesteps = conf.get('timesteps')  # Repeat tensor using this value
        # self.output_dim = conf.get('output_dim')  # Desired output dimension of sequence
        fc_kwargs = conf.fc_layer
        self.out = nn.Linear(**fc_kwargs)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']
        x = x.unsqueeze(1)
        x = x.repeat(1, self.timesteps, 1)
        for model in self.model_list:
            x, _ = model(x)
        x = self.out(x)
        x = self.activation(x)
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyRGCN(ModelBuilder):
    def __init__(self, conf, idx):
        super(PyRGCN, self).__init__()
        rnn_list = []
        gnn_list = []
        self.timesteps = conf.timesteps

        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            if layer+1 % 2 == 1:
                gnn_list.append(tg.ChebConv(**kwargs))
            else:
                rnn_list.append(nn.LSTM(**kwargs, batch_first=True))

        self.gnn_list = nn.ModuleList(gnn_list)
        self.rnn_list = nn.ModuleList(rnn_list)

        fc_kwargs = conf.fc_layer
        self.fc = nn.Linear(**fc_kwargs)

        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)

    def forward(self, x_dict):
        x, adj = x_dict['input'], x_dict['adj']

        for gnn, rnn in zip(self.gnn_list, self.rnn_list):
            out_list = []
            for _ in range(self.timesteps):
                gnn_out = gnn(x, adj)
                out_list.append(gnn_out)
            out_tensor = torch.stack(out_list, 1)
            x, _ = rnn(out_tensor)

        x = x[:, -1, :]  # Take the last timestep's output
        x = self.fc(x)
        x = self.activation(x)
        # x = {'input': x, 'adj': adj}
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class PyCDGM(ModelBuilder):
    def __init__(self, conf, idx):
        super(PyCDGM, self).__init__()
        model_list = []
        gnn_list = []
        self.num_layers = conf.layers

        # Last layer will always be used for GNN weights
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)
            model_list.append(nn.Linear(**kwargs))
            # else:
            #     gnn_list.append(nn.Linear(**kwargs))

        self.model_list = nn.ModuleList(model_list)
        # self.gnn_list = nn.ModuleList(gnn_list)
        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)
        self.device = torch.device('cuda:{}'.format(Config.args.GPU_device)) if torch.cuda.is_available() else \
            torch.device('cpu')
        # TODO change init of these parameters.
        self.temp = torch.nn.Parameter(torch.tensor(1e-1, requires_grad=True, device=self.device))
        self.theta = torch.nn.Parameter(torch.tensor(1e-1, requires_grad=True, device=self.device))

    def forward(self, x_dict):
        x, adj, adj_wts = x_dict['input'], x_dict['adj'], x_dict.get('adj_wts', None)
        initial_adj = x_dict['adj']
        # for mlp, gnn in zip(self.mlp_list, self.gnn_list):
        for k, model in enumerate(self.model_list):
            if k == 0:
                out_x = model(x)
                out_x = self.activation(out_x)
            elif k == self.num_layers - 1:
                x = model(x)
                # x = self.activation(x)
            else:
                out_x = model(out_x)
                out_x = self.activation(out_x)

        # # adj is in edge_index format convert to N x N
        if adj.shape[0] == 2 and adj.shape[0] != adj.shape[1]:
            adj = torch_geometric.utils.to_dense_adj(adj)

        diff = out_x.unsqueeze(1) - out_x.unsqueeze(0)
        dist = -torch.pow(diff, 2).sum(2)
        temp = 1.0 + self.temp
        theta = 5.0 + self.theta
        #print('== Before normalization', dist)
        #print('temp =>', (temp))
        #print('theta =>', (theta))
        #print('feat matrix mean', (-dist).mean())
        #print('feat matrix std', (-dist).std())
        prob_matrix = temp * dist + theta
        prob_matrix = torch.sigmoid(prob_matrix)
        #print('== Probability matrix', prob_matrix)
        adj = prob_matrix + torch.eye(adj.shape[-1]).to(adj.device)
        diag_inv = torch.diag(adj.sum(-1)).inverse()
        x = torch.mm(diag_inv, torch.mm(adj, x))
        x = self.activation(x)

        # convert dense representation of adjacency matrix to sparse
        # if reconverted:
        adj_sparse = torch.nonzero(adj).t().contiguous()

        adj_flatten = torch.flatten(adj)
        adj_wts = adj_flatten[adj_flatten.nonzero()]

        out_dict = {k: v for k, v in x_dict.items()}

        # flatten the adjacency matrix to update the adj_wts matrix

        out_dict['adj_wts'] = adj_wts.squeeze()
        out_dict['input'] = x
        out_dict['adj'] = adj_sparse
        return out_dict


class LGNN(ModelBuilder):
    def __init__(self, conf, idx):
        super(LGNN, self).__init__()
        mlp_list = []
        gl_list = []
        self.num_layers = conf.layers

        # Last layer will always be used for GNN weights
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)

            if layer == conf.layers-1:
                gl_list.append(tg.ChebConv(**kwargs))
            else:
                gl_list.append(nn.Linear(**kwargs))

        self.gl_list = nn.ModuleList(gl_list)

        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)
        self.device = torch.device('cuda:{}'.format(Config.args.GPU_device)) if torch.cuda.is_available() else \
            torch.device('cpu')

        # self.theta = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, device=self.device))

    def forward(self, x_dict):
        x, adj, adj_wts = x_dict['input'], x_dict['adj'], x_dict.get('adj_wts', None)

        gl_x = x
        # for mlp, gnn in zip(self.mlp_list, self.gl_list):
        for k, model in enumerate(self.gl_list[:-1]):
            gl_x = model(gl_x)
            gl_x = self.activation(gl_x)

        # # adj is in edge_index format convert to N x N
        if adj.shape[0] == 2 and adj.shape[0] != adj.shape[1]:
            adj = torch_geometric.utils.to_dense_adj(adj)

        # diff = gl_x.unsqueeze(1) - gl_x.unsqueeze(0)
        # dist = torch.pow(diff, 2).sum(2)
        # dist = -dist/2.0

        diff = gl_x.unsqueeze(1) - gl_x.unsqueeze(0)
        diff = torch.pow(diff, 2).sum(2)
        mask_diff = diff != 0.0
        dist = -torch.sqrt(diff + torch.finfo(torch.float32).eps)
        dist = dist * mask_diff

        prob_matrix = torch.exp(dist)
        adj = prob_matrix
        edge_idx, edge_wts = torch_geometric.utils.dense_to_sparse(prob_matrix)

        gnn_model = self.gl_list[-1]
        x = gnn_model(x, edge_idx, edge_wts)
        x = self.activation(x)

        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict


class LGAT(ModelBuilder):
    def __init__(self, conf, idx):
        super(LGAT, self).__init__()
        mlp_list = []
        gl_list = []
        self.num_layers = conf.layers

        # Last layer will always be used for GNN weights
        for layer in range(conf.layers):
            kwargs = self.set_kwargs(conf, layer, idx)

            if layer == conf.layers-1:
                gl_list.append(tg.GATConv(**kwargs))
            else:
                gl_list.append(nn.Linear(**kwargs))

        self.gl_list = nn.ModuleList(gl_list)

        self.activation = Config.global_dict['model'][conf.activation.name]
        self.activation = self.activation(**conf.activation)
        self.device = torch.device('cuda:{}'.format(Config.args.GPU_device)) if torch.cuda.is_available() else \
            torch.device('cpu')

        # self.theta = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, device=self.device))

    def forward(self, x_dict):
        x, adj, adj_wts = x_dict['input'], x_dict['adj'], x_dict.get('adj_wts', None)

        gl_x = x
        # for mlp, gnn in zip(self.mlp_list, self.gl_list):
        for k, model in enumerate(self.gl_list[:-1]):
            gl_x = model(gl_x)
            gl_x = self.activation(gl_x)

        # # adj is in edge_index format convert to N x N
        if adj.shape[0] == 2 and adj.shape[0] != adj.shape[1]:
            adj = torch_geometric.utils.to_dense_adj(adj)

        diff = gl_x.unsqueeze(1) - gl_x.unsqueeze(0)
        dist = torch.pow(diff, 2).sum(2)
        dist = -dist/2.0
        prob_matrix = torch.exp(dist)
        adj = prob_matrix
        edge_idx, edge_wts = torch_geometric.utils.dense_to_sparse(prob_matrix)

        gnn_model = self.gl_list[-1]
        x = gnn_model(x, edge_idx)
        x = self.activation(x)

        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['input'] = x
        out_dict['adj'] = adj
        return out_dict

