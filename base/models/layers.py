# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models
#
#
# class CustomLeNet(nn.Module):
#     def __init__(self, args):
#         super(CustomLeNet, self).__init__()
#         self.args = args
#
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)  # (3, 6, 5)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
#         self.fc1 = nn.Linear(64*5*12, 128)  # This needs to be a global average pooling layer to obtain the CAM
#         nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu', args.alpha))
#         self.fc2 = nn.Linear(128, args.nfeat)
#         nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu', args.alpha))
#
#         self.leakyrelu = nn.LeakyReLU(args.alpha)
#
#     def forward(self, x):
#         x = torch.mean(x, 1)
#         x = torch.unsqueeze(x, 1)
#         x = self.leakyrelu(self.conv1(x))
#         x = self.leakyrelu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         features = self.fc2(x)
#         return features
#
#
# class CustomAlexNet(nn.Module):
#     """
#     AlexNet with last classification layer substituted by layer with "out_features" nodes
#     """
#
#     def __init__(self, args):
#         super(CustomAlexNet, self).__init__()
#         self.args = args
#
#         # Define CNN on top of GCN
#         self.model = models.alexnet(pretrained=True)  # needs 3 channels
#
#         # Freeze all parameters in the CNN
#         if args.freeze:
#             for param in self.model.parameters():
#                 param.requires_grad = False
#
#         # Reset classes:
#         list(self.model.children())[-1][-1] = torch.nn.Linear(in_features=4096, out_features=args.nfeat, bias=True)
#         nn.init.xavier_uniform_(list(self.model.children())[-1][-1].weight, gain=nn.init.calculate_gain('leaky_relu',
#                                                                                                         args.alpha))
#
#     def forward(self, x):
#         features = self.model(x)
#         return features
#
#
# class CustomResNet(nn.Module):
#     """
#     ResNet with last classification layer substituted by layer with "out_features" nodes
#     """
#
#     def __init__(self, args):
#         super(CustomResNet, self).__init__()
#
#         # Define CNN on top of GCN
#         if args.cnn_model_type == '18':
#             self.model = models.resnet18(pretrained=True)  # needs 3 channels
#         elif args.cnn_model_type == '34':
#             self.model = models.resnet34(pretrained=True)  # needs 3 channels
#         elif args.cnn_model_type == '50':
#             self.model = models.resnet50(pretrained=True)  # needs 3 channels
#         else:
#             raise ValueError('Given model type unknown!')
#
#         # Freeze all parameters in the CNN
#         if args.freeze:
#             for param in self.model.parameters():
#                 param.requires_grad = False
#             for name, layer in self.model.named_modules():
#                 if isinstance(layer, nn.BatchNorm2d):
#                     layer.eval()
#                     layer.track_running_stats = False
#
#         # Reset classes:
#         if args.cnn_model_type == '18':
#             in_features = self.model.layer4[-1].conv2.out_channels
#         elif args.cnn_model_type == '50':
#             in_features = self.model.layer4[-1].conv3.out_channels
#         else:
#             raise ValueError('cnn_model_type unknown')
#
#         self.model.fc = nn.Linear(in_features=in_features, out_features=args.nfeat, bias=True)
#         nn.init.xavier_uniform_(self.model.fc.weight, gain=nn.init.calculate_gain('leaky_relu', args.alpha))
#
#     def forward(self, x):
#         features = self.model(x)
#         return features
#
#
# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, in_features, out_features, args, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = args.dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = args.alpha
#         self.concat = concat
#         self.W = nn.Parameter(torch.zeros((in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.zeros((2*out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, x, adj):
#         h = torch.mm(x, self.W)
#         N = h.size()[0]
#
#         a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
#
#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, h)
#
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#
#
# class NormalizeMat(nn.Module):
#
#     def __init__(self):
#         super(NormalizeMat, self).__init__()
#
#     def forward(self, x):
#         # Row-normalize sparse matrix
#
#         inter = torch.norm(x, 1, 1)
#         inv = torch.pow(inter, -1)
#         sq_inv = torch.diag(inv)
#         output = torch.mm(sq_inv, x)
#
#         return output
