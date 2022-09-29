import os
import copy

import torch
import omegaconf
from omegaconf import OmegaConf
from torch import nn

from base.models.models import ModelBuilder
from base.utils.configuration import Config


class DynamicNet(ModelBuilder):
    """# TODO"""
    def __init__(self, conf):
        super(DynamicNet, self).__init__()
        # Load default model configuration
        self.yaml_file = conf.args.model.yaml_path
        if os.path.exists(self.yaml_file):
            model_config = OmegaConf.load(self.yaml_file)
            conf.args.model = model_config[list(model_config.keys())[0]]
            conf.args.model_name = self.yaml_file.split('/')[-1].replace('.yaml', '')
        else:
            raise FileNotFoundError('DynamicNet configuration file missing.')

        input_dim = conf.args.input_shape_list  # input shape list
        print("Initial input dim ----- >> " + str(input_dim))
        # Just count pairs of Dynamic and Aggregator block pairs for network depth
        num_depth = conf.args.model.get('num', int(len(conf.args.model.keys())/2))

        Config.args.model.num = num_depth

        self.architecture_list = []
        for depth in range(num_depth):
            dyn_conf = conf.args.model["DynamicBlock{}".format(depth)]
            agg_conf = conf.args.model["AggregatorBlock{}".format(depth)]

            print('Initializing DynamicBlock %s' % depth)
            print("----------------DYN INPUT dim ---------------"+ str(depth) + "   " + str(input_dim))
            dyn_block = DynamicBlock(input_dim, dyn_conf, depth)
            input_dim = dyn_block.get_output_dim(conf)
            print("---------------DYN OUTPUT dim ----------------"+ str(depth) + "   " + str(input_dim))

            print('Initializing AggregatorBlock %s' % depth)
            print("-----------------AGG INPUT dim ------------------"+ str(depth) + "   " + str(input_dim))
            agg_block = AggregatorBlock(input_dim, agg_conf, depth)
            input_dim = agg_block.get_output_dim(conf)
            print("----------------AGG OUTPUT dim -------------------"+ str(depth) + "   " + str(input_dim))
            self.architecture_list.append(dyn_block)
            self.architecture_list.append(agg_block)

        self.architecture_list = nn.ModuleList(self.architecture_list)

    def forward(self, x):
        # x = [{'input': img, 'adj': adj}, {'input': img, 'adj': adj}, {'input': feat, 'adj': adj}]
        for k, model_block in enumerate(self.architecture_list):
            x = model_block(x)
        return x


class DynamicBlock(nn.Module):
    """# TODO """
    def __init__(self, input_dim, conf, depth):
        super(DynamicBlock, self).__init__()
        if isinstance(conf.distributor, omegaconf.listconfig.ListConfig):
            self.distributor = list(conf.distributor)
        elif conf.distributor is None:
            self.distributor = list(range(len(input_dim)))
        else:
            raise NotImplementedError('Incorrect distributor argument')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.input_dim = input_dim
        self.output_dim = None
        # input_dim = [img, img, feat, img, feat]

        pmodel = ParallelModel(input_dim, self.distributor, conf, depth)
        self.model_list = pmodel.get_model_list()

        # model_list = [CNN, CNN, GCN, CNN, GCN]

    def get_output_dim(self, conf):
        #print("-------------------- DYN OUTPUT DIM ------------------")
        output_dim = []
        dim = 0
        T_in = {}
        for i, model in enumerate(self.model_list):
            cur_input_dim = list(self.input_dim[self.distributor[i]])

            cur_input_dim[0] = 5  # conf.args.batch_size
            T_in['input'] = torch.rand(cur_input_dim).to(self.device)
            adj = torch.rand(5, 5).to(self.device)  # (conf.args.batch_size, conf.args.batch_size)
            T_in['adj'] = adj.nonzero().t().contiguous()
            model = model.to(self.device)
            T_out = model(T_in)
            #print(T_in)
            dim = T_out['input'].shape
            output_dim.append(tuple(dim))
        return output_dim

    # input_list = [img, img, img, feat]
    # input_list = np.[{'input': feat, 'adj': adj1}, {'input': feat, 'adj': adj2}, {'input': feat, 'adj': adj3}]

    def forward(self, x):
        output_list = []
        for i, model in enumerate(self.model_list):
            output = model(x[self.distributor[i]])
            output_list.append(output)
        return output_list


class ParallelModel:
    """# TODO"""
    def __init__(self, input_dim, distributor, conf, depth):
        self.distributor = distributor
        # if isinstance(conf.distributor, omegaconf.listconfig.ListConfig):
        #     self.distributor = list(conf.distributor)
        # elif conf.distributor is None:
        #     self.distributor = range(len(input_dim))
        # else:
        #     raise NotImplementedError('Incorrect distributor argument')

        self.model_list = list(range(len(self.distributor)))
        self.key_list = ['ParallelMLP', 'ParallelGNN', 'ParallelCNN', 'ParallelRNN', 'ParallelTNN',
                         'ParallelImputer1D', 'ParallelImputer2D', 'ParallelImputer3D']

        self.in_name_list = ['in_features', 'in_channels', 'input_size']
        self.out_name_list = ['out_features', 'out_channels', 'hidden_size']
        output_dim = Config.args.target_shape_list

        for key, parallel_type in conf.items():
            if key in self.key_list and not key == 'distributor':
                for _, val in parallel_type.items():
                    if val.order is None:
                        init_val = val
                        for i in self.distributor:
                            val = copy.deepcopy(init_val)
                            in_dim = input_dim[i]
                            if depth == Config.args.model.num - 1:
                                out_dim = output_dim[i]
                            else:
                                out_dim = None
                            self._set_models(i, key, val, in_dim, out_dim)

                    else:
                        init_val = val
                        for idx, j in enumerate(val.order):
                            val = copy.deepcopy(init_val)
                            print("in parallel model------------")
                            print(self.distributor[j])
                            in_dim = input_dim[self.distributor[j]]
                            if depth == Config.args.model.num - 1:
                                out_dim = output_dim[j]
                            else:
                                out_dim = None
                            val = self._set_input_layer(val, in_dim, out_dim)
                            self.model_list[j] = ModelPiece(val, idx)
            elif not key == 'distributor':
                raise ValueError('Key {} in .yaml file is not know'.format(key))


    # def _get_key_list(self, depth):
    #     key_list = []
    #     if depth == 0:
    #         if Config.args.dirs['img'] is not None:
    #             key_list.append('ParallelCNN')
    #         if Config.args.dirs['feat'] is not None:
    #             key_list.append('ParallelMLP')
    #             key_list.append('ParallelGNN')
    #         if Config.args.dirs['seq'] is not None:
    #             key_list.append('ParallelRNN')
    #             key_list.append('ParallelTNN')
    #     else:
    #         key_list = ['ParallelMLP', 'ParallelGNN', 'ParallelCNN', 'ParallelRNN', 'ParallelTNN']

        # return key_list

    def get_model_list(self):
        return nn.ModuleList(self.model_list)

    def _set_models(self, i, key, val, in_dim, out_dim):
        # Neural Networks
        if len(in_dim) == 2 and key == 'ParallelMLP':
            val = self._set_input_layer(val, in_dim, out_dim)
            self.model_list[i] = ModelPiece(val)
        elif len(in_dim) == 2 and key == 'ParallelGNN':
            val = self._set_input_layer(val, in_dim, out_dim)
            self.model_list[i] = ModelPiece(val)
        elif len(in_dim) == 3 and key == 'ParallelRNN':
            val = self._set_input_layer(val, in_dim, out_dim)
            self.model_list[i] = ModelPiece(val)
        elif len(in_dim) == 4 and key == 'ParallelCNN':
            self.model_list[i] = ModelPiece(val)
        
        # Parallel Imputers
        elif len(in_dim) == 2 and key == 'ParallelImputer1D':
            val = self._set_input_layer(val, in_dim, out_dim)
            self.model_list[i] = ModelPiece(val)
        elif len(in_dim) == 3 and key == 'ParallelImputer2D':
            val = self._set_input_layer(val, in_dim, out_dim)
            self.model_list[i] = ModelPiece(val)
        elif len(in_dim) == 4 and key == 'ParallelImputer3D':
            val = self._set_input_layer(val, in_dim, out_dim)
            self.model_list[i] = ModelPiece(val)
        # if len(in_dim) == 5 and key == 'Parallel3DCNN': TODO
        #     self.model_list[i] = ModelPiece(val)

    def _set_input_layer(self, val, in_dim, out_dim):
        for name in self.in_name_list:
            if name in val.layer0:
                if val.layer0[name] is None:
                    val.layer0[name] = in_dim[-1]
        for name in self.out_name_list:
            if name in val['layer{}'.format(val.layers-1)]:
                if val['layer{}'.format(val.layers-1)][name] is None:
                    val['layer{}'.format(val.layers-1)][name] = out_dim
        return val


class ModelPiece(nn.Module):
    """# TODO"""
    def __init__(self, conf, idx=None):
        super(ModelPiece, self).__init__()
        self.model = Config.global_dict['model'][conf.model]
        self.model = self.model(conf, idx)  # Here, the model instance is created

    def forward(self, x):
        x = self.model(x)
        return x


class AggregatorBlock(nn.Module):
    """# TODO"""
    def __init__(self, input_dim, conf, depth):
        super(AggregatorBlock, self).__init__()
        self.conf = conf
        self.input_dim = input_dim
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        output_dim = Config.args.target_shape_list
        if isinstance(conf.distributor, omegaconf.listconfig.ListConfig):
            self.distributor = list(conf.distributor)
        elif conf.distributor is None:
            if conf.ParallelAgg.Agg0.agg in ['Pass', 'Clone']:
                self.distributor = [[i] for i in range(len(input_dim))]
            elif conf.ParallelAgg.Agg0.agg in ['Concat', 'Sum', 'Average', 'Pool']:  # TODO create separate option MatchConcat, ...
                if depth == Config.args.model.num - 2:
                    self.distributor = [list(range(len(input_dim))) for _ in range(len(output_dim))]
                elif depth < Config.args.model.num - 2:
                    self.distributor = [list(range(len(input_dim)))]
                else:
                    raise ValueError('Wrong aggregator for last AggregatorBlock, should be Pass')
            else:
                raise ValueError('Incorrect aggregator {} in yaml file'.format(conf.ParallelAgg.agg))
        else:
            raise NotImplementedError('Incorrect distributor argument')

        # if conf.distributor is None:
        #     self.distributor = [range(len(input_dim))]
        # elif isinstance(conf.distributor, str):
        #     if conf.distributor == 'concat':
        #         self.distributor = [range(len(input_dim))]
        # elif isinstance(conf.distributor, omegaconf.listconfig.ListConfig):
        #     self.distributor = list(conf.distributor)
        # else:
        #     raise NotImplementedError('Incorrect distributor argument')

        pagg = ParallelAggregator(input_dim, self.distributor, conf, depth)
        self.agg_list = pagg.get_agg_list()

    def get_output_dim(self, conf):
        output_dim = []
        T_in = {}
        for i, agg in enumerate(self.agg_list):
            T_in_list = []
            print("self distri " + str(self.distributor[i]))
            for ind in self.distributor[i]:
                cur_input_dim = list(self.input_dim[ind])
                cur_input_dim[0] = 5
                T_in['input'] = torch.rand(cur_input_dim).to(self.device)
                T_in['adj'] = torch.rand(5, 5).to(self.device)  # (conf.args.batch_size, conf.args.batch_size)
                T_in_list.append(copy.deepcopy(T_in))
            T_out = agg(T_in_list)
            dim = T_out['input'].shape
            output_dim.append(tuple(dim))
        return output_dim

    def forward(self, x):
        output_list = []
        for i, agg in enumerate(self.agg_list):
            input_list = []
            for j in self.distributor[i]:
                input_list.append(x[j])
            output = agg(input_list)
            output_list.append(output)
        return output_list


class ParallelAggregator:
    """# TODO"""
    def __init__(self, input_dim, distributor, conf, depth):
        output_dim = Config.args.target_shape_list
        self.distributor = distributor
        # if isinstance(conf.distributor, omegaconf.listconfig.ListConfig):
        #     self.distributor = list(conf.distributor)
        # elif conf.distributor is None:
        #     if conf.ParallelAgg.Agg0.agg == 'Pass':
        #         self.distributor = [[i] for i in range(len(input_dim))]
        #     elif conf.ParallelAgg.Agg0.agg in ['Concat', 'Sum', 'Average']:
        #         if depth == Config.args.model.num - 2:
        #             self.distributor = [[range(len(input_dim))] for k in range(len(output_dim))]
        #         elif depth < Config.args.model.num - 2:
        #             self.distributor = [range(len(input_dim))]
        #         else:
        #             raise ValueError('Wrong aggregator for last AggregatorBlock, should be Pass')
        #     else:
        #         raise ValueError('Incorrect aggregator {} in yaml file'.format(conf.ParallelAgg.agg))
        # else:
        #     raise NotImplementedError('Incorrect distributor argument')

        self.agg_list = list(range(len(self.distributor)))
        self.key_list = ['ParallelAgg']

        for key, val in conf.ParallelAgg.items():
            if val.order is None:
                for i in range(len(self.distributor)):
                    self.agg_list[i] = AggregatorPiece(val)
            else:
                for j in val.order:
                    self.agg_list[j] = AggregatorPiece(val)

    def get_agg_list(self):
        return nn.ModuleList(self.agg_list)


class AggregatorPiece(nn.Module):
    """# TODO"""
    def __init__(self, conf):
        super(AggregatorPiece, self).__init__()
        self.agg = Config.global_dict['model'][conf.agg]
        self.agg = self.agg(conf)

    def forward(self, x):
        x = self.agg(x)
        return x
