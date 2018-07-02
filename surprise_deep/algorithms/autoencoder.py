from .algo_base_deep import AlgoBaseDeep

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Autoencoder(AlgoBaseDeep):

    def __init__(self, model_option, input_dim):
        super().__init__()
        self.option = model_option
        self.train_ds = None
        self.test_ds = None
        self.input_dim = input_dim
        self.layers = []

        if model_option is None:
            raise Exception("No run option")

        self._create_layers()

    def _create_layers(self):
        if self.option.drop_prob > 0:
            self.drop = nn.Dropout(self.option.drop_prob)

        self.layers = [self.input_dim] + self.option.hidden_layers

        # init the encoder weight
        self.encoder_w = nn.ParameterList(
            [nn.Parameter(torch.rand(self.layers[i + 1], self.layers[i])) for i in range(len(self.layers) - 1)]
        )
        for index, w in enumerate(self.encoder_w):
            init.xavier_uniform(w)

        # init the encoder bias
        self.encoder_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        )

        # init the decoder weight
        reverse_layers = list(reversed(self.layers))
        self.decoder_w = nn.ParameterList(
            [nn.Parameter(torch.rand(reverse_layers[i + 1], reverse_layers[i])) for i in range(len(reverse_layers) - 1)]
        )
        for index, w in enumerate(self.decoder_w):
            init.xavier_uniform(w)

        # init the decoder bias
        self.decoder_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(reverse_layers[i + 1])) for i in range(len(reverse_layers) - 1)]
        )

    def encode(self, x):
        for index, w in enumerate(self.encoder_w):
            input = F.linear(input=x, weight=w, bias=self.encoder_bias[index])
            x = self.activation(input=input, kind=self.option.activation)
        # todo: check dropout

    def decode(self, z):
        # todo: check for constrained autoencoder
        for index, w in enumerate(self.decoder_w):
            input = F.linear(input=z, weight=w, bias=self.decoder_bias[index])
            kind = self.option.activation if index != len(self.layers) - 1 else 'none'
            z = self.activation(input=input, kind=kind)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def fit(self, dataset):
        optimizer = self.optimizer(self.option)
        num_epochs = self.option.num_epochs
        self.train()
        for index, mini_batch in enumerate(dataset.get_mini_batch(batch_size=self.option.batch_size)):
            pass



