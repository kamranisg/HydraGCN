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
	# Update configuration accordiging to desired experiments.
	# dataset_list = ['TADPOLE']
	# model_list = ['MLP']

	# MGMC, HydraGNN
	# dataset_list = ['TADPOLEMeanImputed']
	# dataset_list = ['TADPOLE']
	#dataset_list = ['TADPOLE'] #['TADPOLE']  # 'MNIST5KImgExtract', 'MNIST5KImg', 'MNIST20KImg', 'MNIST20KImgExtract'
	dataset_list = ['COVIDiCTCF']
	dataset_list = ['HAM10K']
	#dataset_list = ['MNIST5KImg']
	#dataset_list = ['ChestXray14SingleLabel']
	# model_list = ['InceptionGCN', 'GCN', 'ChebConv', 'FullGAT', 'MLP']
	# model_list = ['GCN', 'ChebConv', 'MLP', 'FullGAT', 'CDGM', 'BatchInceptionGCN', 'CDGM2',
	#               'FullInceptionGAT', 'BatchInceptionGAT' 'GLAdd', 'FullGATMLP', 'BatchGAT']
	# model_list = ['MLP', 'InceptionGCN', 'GCN', 'ChebConv', 'MultiGCN']
	# model_list = ['FullGAT', 'BatchGAT']
	# model_list = ['FullGAT', 'BatchGAT', 'CDGM', 'LGAT', 'LGNN']
	# model_list = ['BaselineMultiTask', 'HydraGATClassifier', 'HydraGNNClassifier', 'HydraCDGMClassifier']
	# model_list = ['BaselineReconstructImageOnly', 'BaselineReconstructSequenceOnly', 'BaselineReconstructFeatOnly']
	# model_list = ['BaselineClassifier', 'HydraGATClassifier', 'HydraGNNClassifier', 'HydraCDGMClassifier']
	# model_list = ['BaselineMultiTask', 'HydraGATMultiTask', 'HydraCDGMMultiTask', 'HydraGNNMultiTask']
	#model_list = ['MGMC', 'FullGAT']  # 'CNN_Extract', 'MLP', 'CNNGAT', 'EndGAT', 'SkipGAT', 'BaselineClassifier','MGMC'
	#model_list = ['MGMC']
	#model_list = ['CNNMGMC']
	model_list = ['Hydra']
	#model_list = ['CNNGAT']
	# 'HydraCDGMMultiTask'
	# model_list = ['HydraGNN', 'HydraGNNMultiTask']
	#model_yaml_dir = '../base/configs/TADPOLE/MGMCTADPOLE.yaml'
	#model_yaml_dir = '../base/configs/TADPOLE/MGMC.yaml'
	#model_yaml_dir = '../base/configs/COVID/MGMC.yaml'
	#model_yaml_dir = '../base/configs/COVID/Hydra_PyCNN.yaml'  # STATUS QUO
	model_yaml_dir = '../base/configs/COVID/Hydra.yaml'  # STATUS QUO
	model_yaml_dir = '../base/configs/HAM10K/Hydra_HAM10K.yaml'

	#supervision_rate_list = [.05, .25, .5, .75]
	supervision_rate_list = [1.0]

	all_exp_list = list(itertools.product(dataset_list, model_list, supervision_rate_list))
	for dataset_i in all_exp_list:
		Config.args.model.name = config_model_name
		dataset_ = dataset_i[0]
		model_ = dataset_i[1]
		supervision_rate = dataset_i[2]

		Config.args.curr_model_name = model_  # To test if "Extract is in the name"

		# Update configuration in default yaml

		update_default_yaml(Config, dataset_, model_, model_yaml_dir, supervision_rate)
		print("FINISH A")
		# Setup dataset folder
		builder_class = DatasetSetupBuilder.get_class(Config)
		print("FINISH B")
		builder_class().setup_data()
		print("FINISH C")
		# Update dirs in dataset setup
		update_dirs(Config, dataset_, model_)
		print("FINISH D")
		# Full or batch-wise dataset
		dataset_class = DatasetBuilder.get_class(Config)
		print("FINISH E")
		dataset = dataset_class(Config)
		print("FINISH F")

		# Create model architecture
		model_class = ModelBuilder.get_class(Config)
		print("FINISH G")
		model = model_class(Config)
		print("FINISH H")

		# Training
		trainer_class = TrainerBuilder.get_class(Config)
		print("FINISH I")
		trainer = trainer_class(dataset, model, Config)
		print("FINISH J")
		trainer.run_kfold_training()


# Helper functions to modify configuration
def update_default_yaml(conf, dataset_name, model_name, model_yaml_dir, supervision_rate):

	# Supervision rate is used during SSL and when saving SSL results
	conf.args.supervision_rate = supervision_rate

	# ---------- TADPOLE dataset ---------- #
	if dataset_name == 'TADPOLEMeanImputed':

		# Change default.yaml config
		conf.args.dataset.name = dataset_name
		conf.args.dataset.edge_criterion = dataset_name + 'EdgeCriterion'
		conf.args.batchtype = 'full'
		conf.args.multigraph = False
		conf.args.meta_columns = [1, 2, 4]
		conf.args.model.yaml_path = model_yaml_dir.format(dataset_name, model_name)

		if model_name == 'MultiGCN':
			conf.args.multigraph = True

		elif model_name in ['MLP', 'MissMLP', 'BatchGAT', 'BatchInceptionGAT']:
			conf.args.batchtype = 'batch'

	elif dataset_name == 'TADPOLE':
		# Change default.yaml config
		conf.args.dataset.name = dataset_name
		conf.args.dataset.edge_criterion = dataset_name + 'EdgeCriterion'
		conf.args.batchtype = 'full'
		conf.args.multigraph = True
		conf.args.meta_columns = [0, 1, 2]
		conf.args.model.yaml_path = model_yaml_dir.format(dataset_name, model_name, dataset_name)

		if model_name == 'HydraGNN':
			conf.args.multigraph = False

	# ---------- MNIST dataset ---------- #
	elif dataset_name in ['MNIST5KImg', 'MNIST20KImg', 'MNIST5KFeat', 'MNIST5KFeatImgSeq', 'MNIST1KFeatImgSeq', 'MNIST5KImgExtract', 'MNIST20KImgExtract']:
		# Change default.yaml config
		conf.args.batch_size = 1024  # 1024
		conf.args.dataset.name = dataset_name
		conf.args.dataset.edge_criterion = 'MNIST' + 'EdgeCriterion'
		conf.args.batchtype = 'batch'  # 'full'
		conf.args.multigraph = False
		conf.args.meta_columns = [0]
		conf.args.model.yaml_path = model_yaml_dir.format(dataset_name, model_name)

		if model_name == 'MultiGCN':
			conf.args.multigraph = True

		elif model_name in ['MLP', 'MissMLP', 'BatchGAT', 'BatchInceptionGAT']:
			conf.args.batchtype = 'batch'

	elif dataset_name in ['ChestXray14SingleLabel']:
		# Change default.yaml config
		conf.args.batch_size = 500  # 1024
		conf.args.dataset.name = dataset_name
		conf.args.dataset.edge_criterion = 'ChestXray14SL' + 'EdgeCriterion'
		conf.args.batchtype = 'batch'  # 'full'
		conf.args.multigraph = False
		conf.args.meta_columns = [1, 2]
		conf.args.model.yaml_path = model_yaml_dir.format(dataset_name, model_name)

		if model_name == 'MultiGCN':
			conf.args.multigraph = True

		elif model_name in ['MLP', 'MissMLP', 'BatchGAT', 'BatchInceptionGAT']:
			conf.args.batchtype = 'batch'

	# ---------- COVID dataset ---------- #
	elif dataset_name in ['COVIDiCTCF']:
		# Change default.yaml config
		conf.args.dataset.name = dataset_name
		conf.args.dataset.edge_criterion = dataset_name + 'EdgeCriterion'
		conf.args.batchtype = 'batch'
		conf.args.multigraph = True
		conf.args.meta_columns = [0, 1, 2]
		conf.args.model.yaml_path = model_yaml_dir.format(dataset_name, model_name, dataset_name)

		# ---------- HAM10K dataset ---------- #
	elif dataset_name in ['HAM10K']:
		# Change default.yaml config
		conf.args.dataset.name = dataset_name
		conf.args.dataset.edge_criterion = dataset_name + 'EdgeCriterion'
		conf.args.batchtype = 'batch'
		conf.args.multigraph = True
		conf.args.meta_columns = [0]
		conf.args.model.yaml_path = model_yaml_dir.format(dataset_name, model_name, dataset_name)

	else:
		raise ValueError('dataset_name unknown {}'.format(dataset_name))


def update_dirs(conf, dataset_name, model_name):

	# ----------HAM10K dataset ---------- #

	if dataset_name == 'HAM10K':
		if model_name in ['Hydra']:
			conf.args.dirs.adjtoinput.feat = [0]
			conf.args.dirs.feat = [conf.args.dirs.feat[0] ]#conf.args.dirs.feat[0]] #, conf.args.dirs.feat[0]]
			conf.args.dirs.targets = [conf.args.dirs.targets[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0]]
									  #conf.args.dirs.feat[0], conf.args.dirs.feat[0]]
									  ##conf.args.dirs.feat[0], conf.args.dirs.feat[0]]

			conf.args.dirs.loss = [
			{'loss': "CrossEntropyLoss", 'wts': 1., 'supervision_rate': conf.args.supervision_rate},
			{'loss': "FrobeniusNorm", 'wts': 0.01}, {'loss': "DirichletNorm", 'wts': 0.001}]
			#{'loss': "FrobeniusNorm", 'wts': 0.01}, {'loss': "DirichletNorm", 'wts': 0.001},
			#{'loss': "FrobeniusNorm", 'wts': 0.01}, {'loss': "DirichletNorm", 'wts': 0.001}]

	# ---------- COVID dataset ---------- #

	elif dataset_name == 'COVIDiCTCF':
		if model_name in ['MGMC']:
			conf.args.dirs.adjtoinput.feat = [0, 1, 2]
			conf.args.dirs.feat = [conf.args.dirs.feat[0], conf.args.dirs.feat[0], conf.args.dirs.feat[0]]
			conf.args.dirs.targets = [conf.args.dirs.targets[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1000.},
								   {'loss': "FrobeniusNorm", 'wts': 100.}, {'loss': "DirichletNorm", 'wts': 100.},
								   {'loss': "FrobeniusNorm", 'wts': 100.}, {'loss': "DirichletNorm", 'wts': 100.},
								   {'loss': "FrobeniusNorm", 'wts': 100.}, {'loss': "DirichletNorm", 'wts': 100.}]

		elif model_name in ['CNNMGMC']:
			conf.args.dirs.adjtoinput.feat = [0, 1, 2]
			conf.args.dirs.feat = [conf.args.dirs.feat[0], conf.args.dirs.feat[0], conf.args.dirs.feat[0]]#, conf.args.dirs.img_paths[0]]
			conf.args.dirs.targets = [conf.args.dirs.targets[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.targets[0]]

			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1000., 'supervision_rate': conf.args.supervision_rate},
								   {'loss': "FrobeniusNorm", 'wts': 100.}, {'loss': "DirichletNorm", 'wts': 100.},
								   {'loss': "FrobeniusNorm", 'wts': 100.}, {'loss': "DirichletNorm", 'wts': 100.},
								   {'loss': "FrobeniusNorm", 'wts': 100.}, {'loss': "DirichletNorm", 'wts': 100.},
								   {'loss': "CrossEntropyLoss", 'wts': 1000., 'supervision_rate': conf.args.supervision_rate}]

		elif model_name in ['Hydra']:
			conf.args.dirs.adjtoinput.feat = [0, 1, 2]
			conf.args.dirs.feat = [conf.args.dirs.feat[0], conf.args.dirs.feat[0], conf.args.dirs.feat[0]]
			conf.args.dirs.targets = [conf.args.dirs.targets[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0]]

			conf.args.dirs.loss = [
				{'loss': "CrossEntropyLoss", 'wts': 1., 'supervision_rate': conf.args.supervision_rate},
				{'loss': "FrobeniusNorm", 'wts': 0.01}, {'loss': "DirichletNorm", 'wts': 0.001},
				{'loss': "FrobeniusNorm", 'wts': 0.01}, {'loss': "DirichletNorm", 'wts': 0.001},
				{'loss': "FrobeniusNorm", 'wts': 0.01}, {'loss': "DirichletNorm", 'wts': 0.001}]


		else:
			raise NotImplementedError('Using {} for {} not implemented'.format(model_name, dataset_name))

	# ---------- TADPOLE dataset ---------- #
	# Mean imputed TADPOLE dataset
	elif dataset_name == 'TADPOLEMeanImputed':
		if model_name in ['GCN', 'ChebConv', 'ChebConv3', 'ChebConv6', 'ChebConv20', 'MLP', 'MissMLP', 'FullGAT',
						  'CDGM',
						  'InceptionGCN',
						  'ResGL', 'CDGM2',
						  'FullInceptionGAT', 'BatchInceptionGAT', 'GLAdd', 'FullGATMLP', 'BatchGAT', 'LGNN', 'LGAT']:
			conf.args.dirs.feat = [conf.args.dirs.feat[0]]
			conf.args.dirs.adjtoinput.feat = [0, 1, 2]
			conf.args.dirs.targets = [conf.args.dirs.targets[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1.0}]
		elif model_name in ['MultiGCN']:
			conf.args.dirs.feat = [conf.args.dirs.feat[0], conf.args.dirs.feat[0], conf.args.dirs.feat[0]]
			conf.args.dirs.adjtoinput.feat = [0, 1, 2]
			conf.args.dirs.targets = [conf.args.dirs.targets[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1.0}]
		else:
			raise NotImplementedError('Using {} for {} not implemented'.format(model_name, dataset_name))

	# Zero imputed TADPOLE dataset
	elif dataset_name == 'TADPOLE':
		if model_name in ['HydraGAT', 'HydraGATMultiTask']:
			conf.args.dirs.adjtoinput.feat = [0, 1, 2]
			conf.args.dirs.targets = [conf.args.dirs.targets[0], conf.args.dirs.feat[0], conf.args.dirs.feat[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1.0/3.0},
								   {'loss': "FrobeniusNorm", 'wts': 1.0/3.0},
								   {'loss': "DirichletNorm", 'wts': 1.0/3.0}]

		elif model_name in ['MGMC']:
			conf.args.dirs.adjtoinput.feat = [0, 1, 2]
			conf.args.dirs.feat = [conf.args.dirs.feat[-1], conf.args.dirs.feat[-1], conf.args.dirs.feat[-1]]
			conf.args.dirs.targets = [conf.args.dirs.targets[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1000.},
								   {'loss': "FrobeniusNorm", 'wts': 100.}, {'loss': "DirichletNorm", 'wts': 100.},
								   {'loss': "FrobeniusNorm", 'wts': 100.}, {'loss': "DirichletNorm", 'wts': 100.},
								   {'loss': "FrobeniusNorm", 'wts': 100.}, {'loss': "DirichletNorm", 'wts': 100.}]
		elif model_name in ['FullGAT']:
			conf.args.dirs.feat = [conf.args.dirs.feat[0],conf.args.dirs.feat[0],conf.args.dirs.feat[0]]
			conf.args.dirs.adjtoinput.feat = [0, 1, 2]
			conf.args.dirs.targets = [conf.args.dirs.targets[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0], conf.args.dirs.feat[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1.0},
								   {'loss': "FrobeniusNorm", 'wts': 1.0}, {'loss': "DirichletNorm", 'wts': 1.0},
								   {'loss': "FrobeniusNorm", 'wts': 1.0}, {'loss': "DirichletNorm", 'wts': 1.0},
								   {'loss': "FrobeniusNorm", 'wts': 1.0}, {'loss': "DirichletNorm", 'wts': 1.0}]
		else:
			raise NotImplementedError('Using {} for {} not implemented'.format(model_name, dataset_name))

	# ---------- MNIST dataset ---------- #
	elif dataset_name in ['MNIST5KImg', 'MNIST20KImg', 'MNIST5KFeat', 'MNIST5KFeatImgSeq', 'MNIST1KFeatImgSeq', 'MNIST5KImgExtract', 'MNIST20KImgExtract']:
		if model_name in ['GCN', 'ChebConv', 'ChebConv3', 'ChebConv6', 'ChebConv20', 'MissMLP', 'FullGAT',
						  'CDGM', 'InceptionGCN', 'ResGL', 'CDGM2', 'FullInceptionGAT', 'BatchInceptionGAT', 'GLAdd',
						  'FullGATMLP', 'BatchGAT', 'LGNN', 'LGAT']:
			conf.args.dirs.feat = [conf.args.dirs.feat[0]]
			conf.args.dirs.adjtoinput.feat = [0]
			conf.args.dirs.targets = [conf.args.dirs.targets[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1.0}]

		elif model_name in ['CNNGAT', 'CNN_Extract', 'MLP', 'SkipGAT', 'GAT', 'EndGAT']:
			# conf.args.dirs.feat = [conf.args.dirs.feat[0]]
			# conf.args.dirs.adjtoinput.feat = [0]
			# conf.args.dirs.targets = [conf.args.dirs.targets[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1.0, 'supervision_rate':
				conf.args.supervision_rate}]

		elif model_name in ['HydraGATMultiTask', 'HydraGNNMultiTask', 'HydraCDGMMultiTask', 'BaselineMultiTask']:
			conf.args.dirs.targets = [conf.args.dirs.targets[0], conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1.0, 'supervision_rate':
				conf.args.supervision_rate},
								   {'loss': "MSELoss", 'wts': 0.2, 'keyname': 'input', 'target_keyname':
									   'target_output'},
								   {'loss': "MSELoss", 'wts': 0.2, 'keyname': 'input', 'target_keyname':
									   'target_output'},
								   {'loss': "MSELoss", 'wts': 0.2, 'keyname': 'input', 'target_keyname':
									   'target_output'}
								   ]
			conf.args.dirs.metric = [None, {'metric': "ReconstructionMetric"}, {'metric': "ReconstructionMetric"},
									 {'metric': "ReconstructionMetric"}]

		elif model_name in ['HydraGATClassifier', 'HydraGNNClassifier', 'HydraCDGMClassifier', 'BaselineClassifier']:
			# conf.args.dirs.feat = [conf.args.dirs.feat[0]]
			# conf.args.dirs.adjtoinput.feat = [0]
			conf.args.dirs.targets = [conf.args.dirs.targets[0], conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1.0, 'supervision_rate':
				conf.args.supervision_rate},
								   {'loss': "MSELoss", 'wts': .0, 'keyname': 'input', 'target_keyname':
									   'target_output'},
								   {'loss': "MSELoss", 'wts': .0, 'keyname': 'input', 'target_keyname':
									   'target_output'},
								   {'loss': "MSELoss", 'wts': .0, 'keyname': 'input', 'target_keyname':
									   'target_output'}
								   ]
			conf.args.dirs.metric = [None, {'metric': "ReconstructionMetric"}, {'metric': "ReconstructionMetric"},
									 {'metric': "ReconstructionMetric"}]

		########### For debugging, delete later
		elif model_name in ['BaselineReconstructImageOnly']:
			conf.args.dirs.targets = [conf.args.dirs.targets[0], conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 0.0},
								   {'loss': "MSELoss", 'wts': 1.0, 'keyname': 'input', 'target_keyname':
									   'target_output'},
								   {'loss': "MSELoss", 'wts': 0.0, 'keyname': 'input', 'target_keyname':
									   'target_output'},
								   {'loss': "MSELoss", 'wts': 0.0, 'keyname': 'input', 'target_keyname':
									   'target_output'}
								   ]
			conf.args.dirs.metric = [None, {'metric': "ReconstructionMetric"}, {'metric': "ReconstructionMetric"},
									 {'metric': "ReconstructionMetric"}]

		elif model_name in ['BaselineReconstructFeatOnly']:
			conf.args.dirs.targets = [conf.args.dirs.targets[0], conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 0.0},
								   {'loss': "MSELoss", 'wts': 0.0, 'keyname': 'input', 'target_keyname':
									   'target_output'},
								   {'loss': "MSELoss", 'wts': 1.0, 'keyname': 'input', 'target_keyname':
									   'target_output'},
								   {'loss': "MSELoss", 'wts': 0.0, 'keyname': 'input', 'target_keyname':
									   'target_output'}
								   ]
			conf.args.dirs.metric = [None, {'metric': "ReconstructionMetric"}, {'metric': "ReconstructionMetric"},
									 {'metric': "ReconstructionMetric"}]

		elif model_name in ['BaselineReconstructSequenceOnly', 'HydraGATReconstructSequenceOnly']:
			conf.args.dirs.targets = [conf.args.dirs.targets[0], conf.args.dirs.feat[0], conf.args.dirs.feat[0],
									  conf.args.dirs.feat[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 0.0},
								   {'loss': "MSELoss", 'wts': 0.0, 'keyname': 'input', 'target_keyname':
									   'target_output'},
								   {'loss': "MSELoss", 'wts': 0.0, 'keyname': 'input', 'target_keyname':
									   'target_output'},
								   {'loss': "MSELoss", 'wts': 1.0, 'keyname': 'input', 'target_keyname':
									   'target_output'}
								   ]
			conf.args.lr = 0.01
			conf.args.dirs.metric = [None, {'metric': "ReconstructionMetric"}, {'metric': "ReconstructionMetric"},
									 {'metric': "ReconstructionMetric"}]
		########## end debug

		elif model_name in ['CDGMSparse']:
			conf.args.dirs.feat = [conf.args.dirs.feat[0]]
			conf.args.dirs.adjtoinput.feat = [0]
			conf.args.dirs.targets = [conf.args.dirs.targets[0], conf.args.dirs.feat[0]]
			conf.args.dirs.loss = [{'loss': "CrossEntropyLoss", 'wts': 1.0},
								   {'loss': "L1Norm", 'wts': .2, 'keyname': 'adj'}]
			conf.args.dirs.metric = [None,
									 {'metric': "RegularizerMetric"}]

		else:
			raise NotImplementedError('Using {} for {} not implemented'.format(model_name, dataset_name))

	else:
		raise ValueError('{} dataset unknown'.format(dataset_name))
	# NHANES
	# Thyroid
	# ...
if __name__ == "__main__":
	main()
