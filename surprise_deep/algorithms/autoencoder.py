import math
from math import sqrt

from torch.autograd import Variable

from .algo_base_deep import AlgoBaseDeep

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import timeit
import numpy as np
import pandas as pd
import logging


class Autoencoder(AlgoBaseDeep):
    tmp_name = 'training_temp_epoch_'

    def __init__(self, model_option, input_dim):
        super().__init__()
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.option = model_option
        self.logger = self.option.logger(level=logging.INFO)
        self.train_ds = None
        self.test_ds = None
        self.input_dim = input_dim
        self.layers = []
        self.resume_epoch = None
        self.resume_filename = None

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
        self.encoder_w = self.encoder_w.to(device=self.device)

        for index, w in enumerate(self.encoder_w):
            init.xavier_uniform_(w)

        # init the encoder bias
        self.encoder_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        )
        self.encoder_bias = self.encoder_bias.to(device=self.device)

        # init the decoder weight
        reverse_layers = list(reversed(self.layers))
        self.decoder_w = nn.ParameterList(
            [nn.Parameter(torch.rand(reverse_layers[i + 1], reverse_layers[i])) for i in range(len(reverse_layers) - 1)]
        )
        self.decoder_w = self.decoder_w.to(device=self.device)
        for index, w in enumerate(self.decoder_w):
            init.xavier_uniform_(w)

        # init the decoder bias
        self.decoder_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(reverse_layers[i + 1])) for i in range(len(reverse_layers) - 1)]
        )
        self.decoder_bias = self.decoder_bias.to(device=self.device)

    def _resume_training(self):
        if self.option.resume_training:
            temp_files = os.listdir(self.option.get_working_dir())
            for file_name in temp_files:
                if file_name.startswith(self.tmp_name):
                    self.resume_filename = file_name
                    self.load_model(self.resume_filename)
                    self.resume_epoch = self.resume_filename.split('_')[3]

    def encode(self, x):
        for index, w in enumerate(self.encoder_w):
            input = F.linear(input=x, weight=w, bias=self.encoder_bias[index])
            x = self.activation(input=input, kind=self.option.activation)
        if self.option.drop_prob > 0:
            x = self.drop(x)

        return x

    def decode(self, z):
        if self.option.decoder_constraint:
            for index, w in enumerate(
                    list(reversed(self.encoder_w))):  # constrained autoencode re-uses weights from encoder
                input = F.linear(input=z, weight=w.transpose(0, 1), bias=self.decoder_bias[index])
                kind = self.option.activation
                if index == len(self.layers) - 2:
                    if not self.option.last_layer_activations:
                        kind = 'none'
                z = self.activation(input=input, kind=kind)
        else:
            for index, w in enumerate(self.decoder_w):
                input = F.linear(input=z, weight=w, bias=self.decoder_bias[index])
                kind = self.option.activation
                if index == len(self.layers) - 2:
                    if not self.option.last_layer_activations:
                        kind = 'none'
                z = self.activation(input=input, kind=kind)

        return z

    def forward(self, x):
        return self.decode(self.encode(x))

    def learn(self, train_ds):
        self._resume_training()
        optimizer = self.optimizer(self.option)
        num_epochs = self.option.num_epochs
        total_loss = 0
        total_loss_denom = 0.0

        resume_e = 0
        if self.resume_epoch is not None:
            resume_e = int(self.resume_epoch)
            self.resume_epoch = None
        start_time = timeit.default_timer()

        dp = nn.Dropout(p=self.option.noise_prob)

        for epoch in range(resume_e, num_epochs):
            self.train()
            for index, (sparse_row_index, sparse_column_index, sparse_rating, mini_batch) in enumerate(
                    train_ds.get_mini_batch(batch_size=self.option.train_batch_size, input_dim=self.input_dim,
                                            is_normalize=self.option.normalize_data)):
                optimizer.zero_grad()
                inputs = mini_batch.to_dense().to(device=self.device)
                outputs = self.forward(inputs)
                loss = self.MMSEloss(outputs, inputs)
                if math.isnan(loss):
                    continue #todo:check nan loss
                self.logger.debug_(f'epoch {epoch} - loss:{loss}')
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_loss_denom += 1

                if self.option.aug_step > 0:
                    # Magic data augmentation trick happen here
                    for t in range(self.option.aug_step):
                        inputs = Variable(outputs.data)
                        if self.option.noise_prob > 0.0:
                            inputs = dp(inputs)
                        optimizer.zero_grad()
                        outputs = self.forward(inputs)
                        loss = self.MMSEloss(outputs, inputs)
                        loss.backward()
                        optimizer.step()

            self.logger.debug(
                f'epoch:{epoch}, RMSE:{sqrt(total_loss/total_loss_denom)}, total_lost:{total_loss}, denom:{total_loss_denom}')
            self._save_tmp_model(epoch)

        stop_time = timeit.default_timer()
        self.logger.info(f'Learn in: {stop_time-start_time} seconds')
        self._remove_tmp_model(num_epochs)

    def evaluate(self, eval_ds, infer_name):
        self.eval()
        infer_file = os.path.join(self.option.root_dir, self.option.save_dir, infer_name)
        with open(infer_file, 'w') as infer_f:
            first_column_name = eval_ds.option.rating_columns[eval_ds.option.pivot_indexes[0]]
            second_column_name = eval_ds.option.rating_columns[eval_ds.option.pivot_indexes[1]]
            mean_df = None
            if self.option.normalize_data:
                # todo:hardcode code
                if first_column_name == 'userId':
                    mean_df = pd.read_csv(os.path.join(eval_ds.option.get_working_dir(), 'map', 'map_user_mean.csv'))
                elif first_column_name == 'movieId':
                    mean_df = pd.read_csv(os.path.join(eval_ds.option.get_working_dir(), 'map', 'map_movie_mean.csv'))

            third_column_name = 'actual_rating'
            fourth_column_name = 'infer_rating'
            infer_f.write(f'{first_column_name},{second_column_name},{third_column_name},{fourth_column_name}\n')
            for index, (sparse_row_index, sparse_column_index, sparse_rating, mini_batch) in enumerate(
                    eval_ds.get_mini_batch(batch_size=self.option.test_batch_size,
                                           input_dim=self.input_dim,
                                           test_masking_rate=self.option.test_masking_rate)):
                inputs = mini_batch.to_dense().to(device=self.device)
                outputs = self.forward(inputs)
                assert (inputs.shape[0] == outputs.shape[0])
                assert (inputs.shape[1] == outputs.shape[1])
                for i in range(len(sparse_row_index)):
                    predict_value = outputs[sparse_row_index[i], sparse_column_index[i]].item()
                    if self.option.normalize_data:
                        mean = mean_df['mean'][sparse_row_index[i]]
                        predict_value += mean

                    if self.option.prediction_floor:
                        if math.isnan(predict_value):
                            predict_value = 2.5
                        elif predict_value < 0:
                            predict_value = 0
                        elif predict_value > 5:
                            predict_value = 5
                        else:
                            predict_value = round(predict_value * 2)/2

                    self.logger.info_(
                        f'predict row {sparse_row_index[i]} and column {sparse_column_index[i]}:{predict_value}')
                    infer_f.write(f'{sparse_row_index[i]},'
                                  f'{sparse_column_index[i]},'
                                  f'{sparse_rating[i]},'
                                  f'{predict_value}\n')

    def cal_RMSE(self, infer_name):
        pred_file = os.path.join(self.option.root_dir, self.option.save_dir, infer_name)
        with open(pred_file, 'r') as f:
            lines = f.readlines()
            count = 0
            denominator = 0
            for line in lines:
                try:
                    parts = line.split(',')
                    pred = float(parts[3])
                    actual = float(parts[2])
                    denominator += (pred - actual) * (pred - actual)
                    count += 1
                except Exception as e:
                    continue

        rmse = sqrt(denominator / count)
        self.logger.info(f'RMSE {rmse}')
        return rmse

    def _save_tmp_model(self, epoch):
        temp_name = f'{self.tmp_name}{epoch}'
        self.save_model(temp_name)
        if self.resume_filename is not None:
            old_resume_file = os.path.join(self.option.get_working_dir(), self.resume_filename)
            if os.path.exists(old_resume_file):
                os.unlink(old_resume_file)
        self.resume_filename = temp_name

    def _remove_tmp_model(self, num_epochs):
        resume_file = os.path.join(self.option.get_working_dir(), self.resume_filename)
        os.unlink(resume_file)

    def save_model(self, name):
        save_file = os.path.join(self.option.get_working_dir(), f'{name}')
        torch.save(self.state_dict(), save_file)

    def load_model(self, name):
        try:
            load_file = os.path.join(self.option.get_working_dir(), f'{name}')
            self.logger.info(f'load model:{load_file}')
            self.load_state_dict(torch.load(load_file))
            return True
        except Exception as e:
            return False
