import sys
import time
import random
import json

sys.path.append('../..')
sys.path.append('..')
print(sys.path)

default_attrs = {
    'g_root_dir': 'root_dir',
    'g_save_dir': 'save_dir',
    'g_file_name': 'model_option.json',
    'g_force_new_option': False,  # force new option, override current option file
    'm_resume_training': False,
    'mp_learning_rate': 0.005,
    'mp_weight_decay': 0,
    'mp_drop_prob': 0.0,  # dropout drop probability
    'mp_noise_prob': 0.0,
    'mp_train_batch_size': 64,
    'mp_test_batch_size': 1,
    'mp_test_masking_rate': 0.5,
    'mp_num_epochs': 10,
    'mp_optimizer': 'adam',
    'mp_activation': 'selu',  # selu, relu6, etc
    'mp_hidden_layers': [128],
    'mp_aug_step': 1,
    'mp_aug_step_floor': False,
    'mp_decoder_constraint': False,  # reuse weight from the encoder if True
    'mp_normalize_data': True,
    'mp_last_layer_activations': True,
    'mp_prediction_floor': False,  # round off number  to near whole or half, e.g prediction is 1.65 -> 1.5
    'mp_loss_size_average': False,  # when using MMSE loss, use size_average option
    'dp_test_split_rate': [0.1, 0.2, 0.3, 0.4],
    'dp_pivot_indexes': [0, 1],
    '_RMSE': 0,
    '_learn_time': 0,
    '_model_json': ''
}

param_options = {
    'mp_learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.1],
    'mp_weight_decay': [0, 0.0001, 0.001, 0.005, 0.01, 0.1],
    'mp_drop_prob': [0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0],  # dropout drop probability
    'mp_noise_prob': [0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0],
    'mp_train_batch_size': [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
    'mp_test_batch_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'mp_test_masking_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'mp_num_epochs': [1, 2, 4, 8, 10],
    'mp_optimizer': ['adam', 'adagrad', 'rmsprop', 'sgd'],
    'mp_activation': ['selu', 'relu', 'relu6', 'elu', 'lrelu', 'sigmoid', 'tanh', 'swish'],  # selu, relu6, etc
    'mp_hidden_layers': [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2],
    'mp_aug_step': [0, 1, 2, 3, 0, 1, 0],
    'mp_aug_step_floor': [True, False],
    'mp_decoder_constraint': [True, False],  # reuse weight from the encoder if True
    'mp_normalize_data': [True, False],
    'mp_last_layer_activations': [True, False],
    'mp_prediction_floor': [True, False],  # round off number  to near whole or half, e.g prediction is 1.65 -> 1.5
    'mp_loss_size_average': [True, False],  # when using MMSE loss, use size_average option
    'dp_test_split_rate': [0.1, 0.2, 0.3, 0.4],
    'dp_pivot_indexes': [[0, 1], [1, 0]],

    '_RMSE': 0,
    '_learn_time': 0,
    '_model_json': ''
}


def get_default_options():
    ds_option = DatasetOption(g_root_dir='ds_movielens', g_save_dir='100k_alloption_singlerun', d_ds_name='100k')
    model_option = ModelOption(g_root_dir="recsys_deeplearning", g_save_dir="autoencoder_100k_alloption_singlerun")
    return ds_option, model_option


def get_ds_model(ds_option, model_option):
    ds_train = Movielens(ds_option)
    ds_train.download_and_process_data()
    ds_test = Movielens(ds_option, train=False)

    model = Autoencoder(model_option,
                        input_dim=ds_option.d_rating_columns_unique_count[
                            ds_option.dp_pivot_indexes[1]])
    return (ds_train, ds_test, model)


def write_msg(ds_option, model_option, **kwargs):
    msg = ''
    for key, value in model_option.items():
        msg += str(model_option[key]) + ','

    for key, values in param_options.items():
        if ds_option.get(key, None) != None:
            msg += str(ds_option[key]) + ','

    for key, value in kwargs.items():
        msg += str(f'{value},')

    msg = msg.replace('[', '"[')
    msg = msg.replace(']', ']"')

    msg += f'""{json.dumps(model_option)}""'

    recorder.write_line(msg)


def execute(ds_option, model_option):
    (ds_train, ds_test, model) = get_ds_model(ds_option, model_option)
    start_time = time.time()
    model.learn(ds_train)
    learn_time = time.time() - start_time
    model.evaluate(ds_test, 'movielens_100k_preds.txt')
    rmse = model.cal_RMSE("movielens_100k_preds.txt")
    return rmse, learn_time


def run_single():
    for key, values in param_options.items():
        if key.startswith('_'):
            continue

        for value in values:
            (ds_option, model_option) = get_default_options()
            if key == 'mp_hidden_layers':
                model_option[key] = [value]
            else:
                model_option[key] = value
            rmse, learn_time = execute(ds_option, model_option)
            write_msg(ds_option, model_option, rmse=rmse, learn_time=learn_time)


def run_random(repeat=1000):
    all_options = {}
    random_key = ''
    count = 0

    while (count < repeat):
        count += 1
        random_option = {}
        for key, values in param_options.items():
            if key.startswith('_'):
                continue

            random_value = ''
            if key == 'mp_hidden_layers':
                layers_num = [1, 2, 3, 4, 5]
                random_max_layer = random.choice(layers_num)
                layers = []
                for i in range(random_max_layer):
                    layers.append(random.choice(values))

                order = random.choice(['up', 'down', 'random'])
                if order == 'up':
                    layers.sort()
                elif order == 'down':
                    layers.sort(reverse=True)

                random_option[key] = layers
            else:
                random_value = random.choice(values)
                random_option[key] = random_value
            random_key += f'{key}_{str(random_value)}_'

        if all_options.get(random_key, None) != None:
            print('option already executed, skip')
            continue
        else:
            print(f'key:{random_key}\n')
            all_options[random_key] = random_option

        (ds_option, model_option) = get_default_options()
        for key, value in random_option.items():
            if ds_option.get(key, None) != None:
                ds_option[key] = value
            elif model_option.get(key, None) != None:
                model_option[key] = value

        rmse, learn_time = execute(ds_option, model_option)
        write_msg(ds_option, model_option, rmse=rmse, learn_time=learn_time)


if __name__ == '__main__':
    recorder = Recorder(g_root_dir="recsys_deeplearning", g_save_dir="autoencoder_100k_alloption_singlerun")
    recorder.open()
    column_names = ''
    for item in list(default_attrs.keys()):
        column_names += item + ','
    column_names = column_names[0:len(column_names) - 1]
    recorder.write_line(column_names)
    # run_single()
    run_random(repeat=5000)

    recorder.close()
