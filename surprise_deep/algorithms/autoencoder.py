from .algo_base_deep import AlgoBaseDeep

import torch
import torch.nn as nn


class Autoencoder(AlgoBaseDeep):
    num_epochs = 50
    batch_size = 100
    hidden_size = 30

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

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)
        iter_per_epoch = len()
