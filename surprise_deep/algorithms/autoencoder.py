from .algo_base_deep import AlgoBaseDeep

import torch
import torch.nn as nn


class Autoencoder(AlgoBaseDeep):

    def __init__(self, in_dim, hid_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def fit(self, dataset, train_options=None):
        if torch.cuda.is_available():
            self.cuda()

