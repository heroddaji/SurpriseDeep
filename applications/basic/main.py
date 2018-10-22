from surprise import SVD
from surprise import Dataset
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise.model_selection import cross_validate

from drecsys import DataReader
from drecsys import AlgoBaseline
from drecsys import cross_validation

def run_surprise():
    # Load the movielens-100k dataset (download it if needed).
    data = Dataset.load_builtin('ml-100k')

    # Use the famous SVD algorithm.
    algo_svd = SVD()
    algo_normal = NormalPredictor()
    algo_baseline = BaselineOnly()
    algo_knnBasic = KNNBasic()

    # Run 5-fold cross-validation and print results.
    cross_validate(algo_svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    cross_validate(algo_normal, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    cross_validate(algo_baseline, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    cross_validate(algo_knnBasic, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


def run_drecsys():
    data = DataReader("../../datasets/movielens/ml-100k/u.data",delim='\t')
    algo_baseline = AlgoBaseline()
    cross_validation(algo_baseline, data)



if __name__ == '__main__':
    run_surprise()
    run_drecsys()
