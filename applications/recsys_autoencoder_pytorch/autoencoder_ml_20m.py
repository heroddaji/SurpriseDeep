import sys

sys.path.append('../..')
sys.path.append('..')
print(sys.path)

pre_train = False
model_name = 'autoencoder_20m'
predict_name = 'movielens_20m_preds.txt'
dataset = '20m'

if not pre_train:
    ml_option = DatasetOption(root_dir='ds_movielens', ds_name=dataset, save_dir=dataset)
    ml_ds_train = Movielens(ml_option)
    ml_ds_train.download_and_process_data()
    ml_ds_test = Movielens(ml_option, train=False)

    model_option = ModelOption(root_dir="recsys_deeplearning", save_dir=model_name)
    model = Autoencoder(model_option,
                        input_dim=ml_option.rating_columns_unique_count[
                            ml_option.pivot_indexes[1]])  # get the movie count as number of columns
    print(model)
    model.learn(ml_ds_train)
    model.save_model(f'{model_name}.model')
    model.load_model(f'{predict_name}.model')
    model.evaluate(ml_ds_test, f'{predict_name}')
    model.cal_RMSE(f'{predict_name}')
else:
    ml_option = DatasetOption(root_dir='ds_movielens', ds_name=dataset, save_dir=dataset)
    ml_ds_test = Movielens(ml_option, train=False)
    model_option = ModelOption(root_dir="recsys_deeplearning", save_dir=model_name)
    model = Autoencoder(model_option,
                        input_dim=ml_option.rating_columns_unique_count[
                            ml_option.pivot_indexes[1]])  # get the movie count as number of columns

    model.load_model(f'{model_name}.model')
    model.evaluate(ml_ds_test, f'{predict_name}')
    model.cal_RMSE(f'{predict_name}')
# bug: always unzip 20m file
