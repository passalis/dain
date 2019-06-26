from train_utils import train_evaluate_anchored, get_average_metrics
import torch.nn as nn
from dain import DAIN_Layer


class MLP(nn.Module):

    def __init__(self, mode='adaptive_avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.0001):
        super(MLP, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(15 * 144, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 3)
        )

        self.dean = DAIN_Layer(mode=mode, mean_lr=mean_lr, gate_lr=gate_lr, scale_lr=scale_lr)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.dean(x)
        x = x.contiguous().view(x.size(0), 15 * 144)
        x = self.base(x)

        return x


def run_experiments_ablation(model, mode, train_epochs=20, window=10, normalization=None):

    results1 = train_evaluate_anchored(model, window=window, train_epochs=train_epochs, horizon=0,
                                       splits=[1, 2, 3, 4, 5, 6, 7, 8],
                                       normalization=normalization)


    print("----------")
    print("Mode: ", mode)
    metrics_1 = get_average_metrics(results1)
    print(metrics_1)


mean_lr, std_lr, scale_lr = 1e-06, 0.001, 10

model = lambda: MLP(mode=None, mean_lr=mean_lr, gate_lr=scale_lr, scale_lr=std_lr)
run_experiments_ablation(model, 'mlp_std', window=15, normalization='std')

model = lambda: MLP(mode='avg', mean_lr=mean_lr, gate_lr=scale_lr, scale_lr=std_lr)
run_experiments_ablation(model, 'mlp_sample_avg', window=15, normalization=None)

model = lambda: MLP(mode='full', mean_lr=mean_lr, gate_lr=scale_lr, scale_lr=std_lr)
run_experiments_ablation(model, 'mlp_full', window=15, normalization=None)
