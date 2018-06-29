from .algo_base_deep import AlgoBaseDeep

import torch
import torch.nn as nn
import torch.nn.init as init


class Autoencoder(AlgoBaseDeep):

    def __init__(self, model_option, input_dim):
        super().__init__()
        self.option = model_option
        self.train_ds = None
        self.test_ds = None
        self.input_dim = input_dim

        if model_option is None:
            raise Exception("No run option")

        self._create_layers()

    def _create_layers(self):
        if self.option.drop_prob > 0:
            self.drop = nn.Dropout(self.option.drop_prob)

        layers = [self.input_dim] + self.option.hidden_layers

        #init the encoder weight
        self.encoder_w = nn.ParameterList(
            [nn.Parameter(torch.rand(layers[i + 1], layers[i])) for i in range(len(layers) - 1)]
        )
        for index, w in enumerate(self.encoder_w):
            init.xavier_uniform(w)

        # init the encoder bias
        self.encoder_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(layers[i + 1])) for i in range(len(layers) - 1)]
        )

        #init the decoder weight
        reverse_layers = list(reversed(layers))
        self.decoder_w = nn.ParameterList(
            [nn.Parameter(torch.rand(reverse_layers[i + 1], reverse_layers[i])) for i in range(len(reverse_layers) - 1)]
        )
        for index, w in enumerate(self.decoder_w):
            init.xavier_uniform(w)

        # init the decoder bias
        self.decoder_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(reverse_layers[i + 1])) for i in range(len(reverse_layers) - 1)]
        )


    def forward(self, x):
        return self.decoder(self.encoder(x))

    def fit(self, dataset, train_options=None):
        if torch.cuda.is_available():
            self.cuda()
