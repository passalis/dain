import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DAIN_Layer(nn.Module):
    def __init__(self, mode='adaptive_avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144):
        super(DAIN_Layer, self).__init__()
        assert mode in [None, 'avg', 'adaptive_avg', 'adaptive_scale', 'full'], f'Unsupported mode: {mode}!'\
                'Use one of: None, "avg", "adaptive_avg", "adaptive_scale", "full". '
        print("Mode = ", mode)

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average; aka Dain(1) 
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling; Dain(1+2)
        if mode == 'adaptive_scale' or mode == 'full': 
            self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
            self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive gating; Dain(1+2+3)
        if mode == 'full':
            self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)

        ## Other methods:
        # Nothing to normalize
        if self.mode == None:
            return x
        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            avg = avg.resize(avg.size(0), avg.size(1), 1)
            x = x - avg
            return x

        ## DAIN:
        # Perform the first step:   adaptive averaging; DAIN(1)
        # Step 1:
        avg = torch.mean(x, 2)
        adaptive_avg = self.mean_layer(avg)
        adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
        x = x - adaptive_avg
        if self.mode == 'adaptive_avg':
            return x

        # Perform the second step:  adaptive averaging + adaptive scaling; DAIN(1+2)
        # Step 2:
        std = torch.mean(x ** 2, 2)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        adaptive_std[adaptive_std <= self.eps] = 1

        adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
        x = x / adaptive_std
        if self.mode == 'adaptive_scale':
            return x

        # Perform the third step:   adaptuve avg + adative scale + gating; DAIN(1+2+3)
        # Step 3: 
        avg = torch.mean(x, 2)
        gate = F.sigmoid(self.gating_layer(avg))
        gate = gate.resize(gate.size(0), gate.size(1), 1)
        x = x * gate
        if self.mode == 'full':
            return x

        assert False, "You fool! Should not reach here."
