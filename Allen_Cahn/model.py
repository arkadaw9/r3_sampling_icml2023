import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from collections import OrderedDict
import numpy as np

# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers, activation, L=1.0, M=1, use_batch_norm=False, use_instance_norm=False):
        super(DNN, self).__init__()
        
        self.L = L
        self.M = M
        print(f"Initializing a default MLP with L:{L}, M:{M}, layers: {layers}")
        
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
    
    def get_activation(self, activation):
        if activation == 'identity':
            return torch.nn.Identity
        elif activation == 'tanh':
            return torch.nn.Tanh
        elif activation == 'relu':
            return torch.nn.ReLU
        elif activation == 'gelu':
            return torch.nn.GELU
    
    def input_encoding(self, t, x):
        w = 2.0 * torch.pi / self.L
        k = torch.arange(1, self.M + 1, device=x.device)
        out = torch.cat([t, torch.ones_like(t), 
                         torch.cos(k * w * x), torch.sin(k * w * x)], dim=-1)
        return out
    
    def forward(self, inp):
        t = inp[:,0:1]
        x = inp[:,1:2]
        out = self.input_encoding(t=t, x=x)
        out = self.layers(out)
        return out


# the deep neural network
class modified_MLP(torch.nn.Module):
    def __init__(self, layers, activation, M=1, L=1.0, use_batch_norm=False, use_instance_norm=False, init="xavier"):
        super(modified_MLP, self).__init__()
        
        self.L = L
        self.M = M
        print(f"Initializing a modified MLP with L:{L}, M:{M}, layers: {layers}")
        
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
        
        if init=="xavier":
            self.xavier_init_weights()
        elif init=="kaiming":
            self.kaiming_init_weights()

    def input_encoding(self, t, x):
        w = 2.0 * torch.pi / self.L
        k = torch.arange(1, self.M + 1, device=x.device)
        out = torch.cat([t, torch.ones_like(t), 
                         torch.cos(k * w * x), torch.sin(k * w * x)], dim=-1)
        return out
    
    def forward(self, inp):
        t = inp[:,0:1]
        x = inp[:,1:2]
        out = self.input_encoding(t=t, x=x)

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

    def xavier_init_weights(self):
        with torch.no_grad():
            print("Initializing Network with Xavier Initialization..")
            for m in self.layer_list:
                if hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.0)

    def kaiming_init_weights(self):
        with torch.no_grad():
            print("Initializing Network with Kaiming Initialization..")
            for m in self.layer_list:
                if hasattr(m, 'weight'):
                    nn.init.kaiming_uniform_(m.weight)
                    m.bias.data.fill_(0.0)