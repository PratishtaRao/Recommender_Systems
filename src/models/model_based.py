import pandas as pd
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate
from surprise import NormalPredictor, NMF, SlopeOne, CoClustering
from surprise import KNNBasic, KNNWithZScore, KNNWithMeans, KNNBaseline
from surprise import BaselineOnly, SVD,SVDpp
from surprise.accuracy import rmse
from surprise.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from collections import defaultdict


def basic_model_based(train, test, algo):
    algorithm = algo
    algorithm.fit(train)
    predictions = algorithm.test(test)
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)
    precision, recall = sum(prec for prec in precisions.values()) / len(precisions),sum(rec for rec in recalls.values()) / len(recalls)
    f1score = 2*((precision*recall)/(precision+recall))
    return rmse(predictions), precision, recall, f1score


def plot_for_rmse(train, test):
    error_basic = []
    error_baseline =[]
    error_means = []
    error_zscore = []
    number = []
    for num in range(1, 25):
        number.append(num)
        rmse_basic = basic_model_based(train, test,
                                       KNNBasic(k=num, sim_options=similarity_measure('pearson', 1)))
        rmse_basic = float("{:.3f}".format(rmse_basic))
        error_basic.append(rmse_basic)

        rmse_baseline = basic_model_based(train, test,
                                       KNNBaseline(k=num, sim_options=similarity_measure('pearson', 1)))
        rmse_baseline = float("{:.3f}".format(rmse_baseline))
        error_baseline.append(rmse_baseline)

        rmse_means = basic_model_based(train, test,
                                       KNNWithMeans(k=num, sim_options=similarity_measure('pearson', 1)))
        rmse_means = float("{:.3f}".format(rmse_means))
        error_means.append(rmse_means)

        rmse_zscore = basic_model_based(train, test,
                                       KNNWithZScore(k=num, sim_options=similarity_measure('pearson', 1)))
        rmse_zscore = float("{:.3f}".format(rmse_zscore))
        error_zscore.append(rmse_zscore)

    plt.plot(number, error_basic, marker='o', markerfacecolor='blue', markersize=5, label= 'Basic')
    plt.plot(number, error_baseline, marker='', color='olive', linewidth=2, label = 'Baseline')
    plt.plot(number, error_means, marker=' ', color='black', linewidth=2, linestyle='dashed', label='Means')
    plt.plot(number, error_zscore, marker='', color='orchid', linewidth=2, linestyle = 'dashdot',label= 'ZScore')
    plt.legend()
    plt.xlabel('Value of K')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Plot for K vs RMSE')
    plt.show()


def similarity_measure(name, flag):
    sim_options = {}
    if flag == 0:
        sim_options = {'name': name,
                       'user_based': False
                       }
    elif flag == 1:
        sim_options = {'name': name
                        }
    else:
        print('Please entry valid number for flag parameter')

    return sim_options


def precision_recall_at_k(predictions, k=10, threshold=3.5):

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


def crossvalidate(data):
    results = []
    for algorithm in [NormalPredictor(), KNNBaseline(k=15, sim_options=similarity_measure('pearson', 1)),
                      KNNBasic(k=15, sim_options=similarity_measure('pearson', 1)),
                      KNNWithMeans(k=15, sim_options=similarity_measure('pearson', 1)),
                      KNNWithZScore(k=15, sim_options=similarity_measure('pearson', 1)), BaselineOnly(), SVD(),
                      SVDpp(), NMF(), SlopeOne(), CoClustering()]:
        result = cross_validate(algorithm, data, measures=['RMSE'], cv=5, verbose=False)
        temp = pd.DataFrame.from_dict(result).mean(axis=0)
        temp = temp.append(pd.Series([str(algorithm).split(' ')[0].split(".")[-1]], index=['Algorithm']))
        results.append(temp)
    rmse_values = pd.DataFrame(results).set_index('Algorithm').sort_values('test_rmse')
    return rmse_values


def gridsearch(data, algo, param_grid):
    # param_grid = {'n_factors': [50, 100, 150], 'n_epochs': [20, 30],
    #               'lr_all': [0.005, 0.01], 'reg_all': [0.02, 0.1]}

    gs = GridSearchCV(algo, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)
    params = gs.best_params['rmse']
    print(params)
    #
    # svdtuned = SVD(reg_all=params['reg_all'], n_factors=params['n_factors'], n_epochs=params['n_epochs'],
    #                lr_all=params['lr_all'])


def main():
    book_df = pd.read_csv("../../data/processed/filtered_ratings.csv")
    # Reader object and rating scale specification
    book_df = book_df.drop('Unnamed: 0', axis=1)
    reader = Reader(rating_scale=(1, 5))
    # Load data
    data = Dataset.load_from_df(book_df[["user_id", "book_id", "rating"]], reader)

    # Spilt data into train and test sets
    train_set, test_set = train_test_split(data, test_size=0.20)

    algorithm_list = [NormalPredictor(), BaselineOnly(), KNNWithZScore(k=10, sim_options=similarity_measure('pearson', 1)),
                      KNNWithMeans(k=10, sim_options=similarity_measure('pearson', 1)), KNNBaseline(k=10, sim_options=similarity_measure('pearson', 1)),
                      KNNBasic(k=10, sim_options=similarity_measure('pearson', 1)), SVDpp(), SVD(), NMF()]

    # # Fit model for normal predictor and get rmse
    # basic_model_based(train_set, test_set, NormalPredictor())
    #
    # # Fit model for Baselineonly algorithm
    # basic_model_based(train_set, test_set, BaselineOnly())
    #
    # # Fit model for KNN algorithms
    # basic_model_based(train_set, test_set, KNNBasic(k=10, sim_options=similarity_measure('pearson', 1)))
    #
    # plot_for_rmse(train_set, test_set)
    # Crossvalidation results
    # res = crossvalidate(data)
    # print(res)
    results = {}
    for algo in algorithm_list:
        rmse, preci, recall, f1 = basic_model_based(train_set, test_set, algo)
        print("Algorithm:", algo, preci, recall, f1)
        print("**------------------------------------------------------------------------------------------**")


if __name__ == '__main__':
    main()


