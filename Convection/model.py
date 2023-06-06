import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from collections import OrderedDict
import numpy as np

    
# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers, activation, use_batch_norm=False, use_instance_norm=False, init="default"):
        super(DNN, self).__init__()
        
        print(f"Initializing a default MLP with layers: {layers}")
        
        # parameters
        self.depth = len(layers) - 1
        
        self.activation = self.get_activation(activation)
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, nn.Linear(layers[i], layers[i+1]))
            )

            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i+1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i+1])))

            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        self.init_weights(init=init)
        
            
    def get_activation(self, activation):
        if activation == 'identity':
            return torch.nn.Identity
        elif activation == 'tanh':
            return torch.nn.Tanh
        elif activation == 'relu':
            return torch.nn.ReLU
        elif activation == 'gelu':
            return torch.nn.GELU

    def forward(self, inp):
        out = self.layers(inp)
        return out
    
    def init_weights(self, init):
        if init in ['xavier_uniform','kaiming_uniform', 'xavier_normal', 'kaiming_normal']:
            with torch.no_grad():
                print(f"Initializing Network with {init} Initialization..")
                for m in self.layers:
                    if hasattr(m, 'weight'):
                        if init == "xavier_uniform":
                            nn.init.xavier_uniform_(m.weight)
                        elif init == "kaiming_uniform":
                            nn.init.kaiming_uniform_(m.weight)
                        elif init == "xavier_normal":
                            nn.init.xavier_normal_(m.weight)
                        elif init == "kaiming_normal":
                            nn.init.kaiming_normal_(m.weight)
                        m.bias.data.fill_(0.0)


# the deep neural network
class modified_MLP(torch.nn.Module):
    def __init__(self, layers, activation, use_batch_norm=False, use_instance_norm=False, init="xavier"):
        super(modified_MLP, self).__init__()
        
        print(f"Initializing a modified MLP with layers: {layers}")
        
        # parameters
        self.depth = len(layers) - 1
        
        self.activation = self.get_activation(activation)
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm
        
        self.layer_U = torch.nn.Linear(layers[0], layers[1])
        self.layer_V = torch.nn.Linear(layers[0], layers[1])
        
        self.layer_list = []
        for i in range(self.depth - 1): 
            self.layer_list.append(
                torch.nn.Linear(layers[i], layers[i+1])
            )
            if self.use_batch_norm:
                layer_list.append(torch.nn.BatchNorm1d(num_features=layers[i+1]))
            if self.use_instance_norm:
                layer_list.append(torch.nn.InstanceNorm1d(num_features=layers[i+1]))
            self.layer_list.append(self.activation())
        self.layer_list.append(
            torch.nn.Linear(layers[-2], layers[-1])
        )
        
        self.layer_list = torch.nn.ModuleList(self.layer_list)
        self.init_weights(init=init)


    
    def forward(self, inp):
        out = inp
        u = self.activation()(self.layer_U(out))
        v = self.activation()(self.layer_V(out))
        
        for layer in self.layer_list[:-1]:
            out = layer(out)
            out = out * u + (1-out) * v
            
        out = self.layer_list[-1](out)
        return out
    
    def get_activation(self, activation):
        if activation == 'identity':
            return torch.nn.Identity
        elif activation == 'tanh':
            return torch.nn.Tanh
        elif activation == 'relu':
            return torch.nn.ReLU
        elif activation == 'gelu':
            return torch.nn.GELU
    
    def init_weights(self, init):
        if init in ['xavier_uniform','kaiming_uniform', 'xavier_normal', 'kaiming_normal']:
            with torch.no_grad():
                print(f"Initializing Network with {init} Initialization..")
                for m in self.layer_list:
                    if hasattr(m, 'weight'):
                        if init == "xavier_uniform":
                            nn.init.xavier_uniform_(m.weight)
                        elif init == "kaiming_uniform":
                            nn.init.kaiming_uniform_(m.weight)
                        elif init == "xavier_normal":
                            nn.init.xavier_normal_(m.weight)
                        elif init == "kaiming_normal":
                            nn.init.kaiming_normal_(m.weight)
                        m.bias.data.fill_(0.0)
