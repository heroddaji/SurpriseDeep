def cross_validate(algo, data, measures=['rmse', 'mae'], cv=None,
                   return_train_measures=False, n_jobs=1,
                   pre_dispatch='2*n_jobs', verbose=False):
    print('hehe')