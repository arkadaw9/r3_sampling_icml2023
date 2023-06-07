import torch
import torch.nn as nn


# the deep neural network
class modified_MLP(nn.Module):
    def __init__(self, layers, activation, use_batch_norm=False, use_instance_norm=False, init="xavier", verbose=False):
        super(modified_MLP, self).__init__()
        if verbose:
            print(f"Initializing a modified MLP with layers: {layers}")
            
        # parameters
        self.depth = len(layers) - 1
        self.verbose = verbose
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm
        
        self.layer_U = nn.Linear(layers[0], layers[1])
        self.layer_V = nn.Linear(layers[0], layers[1])
        
        self.layer_list = []
        for i in range(self.depth - 1): 
            self.layer_list.append(
                nn.Linear(layers[i], layers[i+1])
            )
            if self.use_batch_norm:
                layer_list.append(nn.BatchNorm1d(num_features=layers[i+1]))
            if self.use_instance_norm:
                layer_list.append(nn.InstanceNorm1d(num_features=layers[i+1]))
            self.layer_list.append(self.activation())
        self.layer_list.append(
            nn.Linear(layers[-2], layers[-1])
        )
        
        self.layer_list = nn.ModuleList(self.layer_list)
        
        if init=="xavier":
            self.xavier_init_weights()
        elif init=="kaiming":
            self.kaiming_init_weights()
    
    def forward(self, inp):
        out = inp
        u = self.activation()(self.layer_U(out))
        v = self.activation()(self.layer_V(out))
        
        for layer in self.layer_list[:-1]:
            out = layer(out)
            out = out * u + (1-out) * v
            
        out = self.layer_list[-1](out)
        return out

    def xavier_init_weights(self):
        with torch.no_grad():
            if self.verbose:
                print("Initializing Network with Xavier Initialization..")
            for m in self.layer_list:
                if hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.0)

    def kaiming_init_weights(self):
        with torch.no_grad():
            if self.verbose:
                print("Initializing Network with Kaiming Initialization..")
            for m in self.layer_list:
                if hasattr(m, 'weight'):
                    nn.init.kaiming_uniform_(m.weight)
                    m.bias.data.fill_(0.0)
