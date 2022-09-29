from setups.setup_data import data_setup
from base.datasets.datasets_types import get_dataset_type
import argparse
from base.trainer.train_base import get_trainer_class

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNISTFeat', help='Specifies the used dataset.')
parser.add_argument('--datatype', type=str, default='full', help='Specifies the used type of dataset.')
parser.add_argument('--train_mode', type=str, default='classification', help='Specifies the used type of training.')
parser.add_argument('--folds', type=int, default=23, help='Specifies the number of folds.')
parser.add_argument('--meta_columns', type=int, nargs='+', default=[0], help='Used columns of meta data')
parser.add_argument('--multigraph', type=bool, default=False, help='Defines if multigraph setting is used')
parser.add_argument('--model', type=str, default='PyGAT', help='Specifies the used type of model.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs.')
parser.add_argument('--patience', type=int, default=30, help='Patience until training is stopped with no improvement.')
parser.add_argument('--num_print', type=int, default=10, help='Steps between printed epochs.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--GPU_device', type=str, default='0', help='Specifies the used GPU.')
parser.add_argument('--writing', type=bool, default=False, help='Define if writer for Tensorboard should be used.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=200, help='Batch size.')
parser.add_argument('--label_list', type=str, nargs='+', default=[0, 1, 2], help='Used labels,'
                                                                                       'put as "--label_list "3" "5" .')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nfeat', type=int, default=392, help='Number of features used as input for GAT.')
parser.add_argument('--ngout', type=int, default=10, help='Dimension of GAT output.')
parser.add_argument('--nhid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nheads', type=int, default=5, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.3, help='Alpha for the leaky_relu.')
parser.add_argument('--gnn_layers', type=int, nargs='+', default=[392, 64, 10], help='Dimensions of input layer, '
                                                                               'hidden layer/s and output layer as a '
                                                                               'list.')
args = parser.parse_args()

# --------------------------------------------------------
# Setup dataset folder
dirs = data_setup(args.dataset)
# --------------------------------------------------------

dataset_class = get_dataset_type(args.datatype)
dataset = dataset_class(dirs, args)

# --------------------------------------------------------
# Start the training
trainer_class = get_trainer_class(args)
Trainer = trainer_class(dataset, args)
Trainer.run_kfold_training()
