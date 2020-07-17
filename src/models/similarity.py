import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def create_matrix(data):
    data = data.drop_duplicates(['user_id', 'book_id'])
    user_item_mat = data.pivot(index='user_id', columns='book_id', values='rating')
    user_item_mat = user_item_mat.fillna(0)
    user_matrix = user_item_mat.to_numpy()

    # Check Sparsity
    non_sparsity = float(len(user_matrix.nonzero()[0]))
    non_sparsity /= (user_matrix.shape[0] * user_matrix.shape[1])
    non_sparsity *= 100
    print('Sparsity: {:4.2f}%'.format(non_sparsity))
    return user_matrix


def create_split(data):
    test = np.zeros(data.shape)
    train = data.copy()
    for row in range(data.shape[0]):
        test_data = np.random.choice(data[row, :].nonzero()[0],
                                     size=10,
                                     replace=False)
        train[row, test_data] = 0
        test[row, test_data] = data[row, test_data]

        return train, test


def cosine_similarity(data, type='user', epsilon= 1e-9):
    if type == 'user':
        sim = data.dot(data.T) + epsilon
    elif type == 'item':
        sim = data.T.dot(data) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


def predict_rating(data, sim, type='user'):
    if type == 'user':
        return sim.dot(data) / np.array([np.abs(sim).sum(axis=1)]).T
    elif type == 'item':
        return data.dot(sim) / np.array([np.abs(sim).sum(axis=1)])


def evaluation(pred, actual):
    # Ignore nonzero values
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


def top_k_recommendations(similarity, idx, k=6):
    matches = np.argsort(similarity[idx, :])[:-k-1:-1]


def main():
    ratings = pd.read_csv("../data/processed/filtered_ratings.csv")
    ratings = ratings.drop(['Unnamed: 0'], axis=1)
    # Create user - item matrix
    user_item_matrix = create_matrix(ratings)
    # Create train and test split
    train, test = create_split(user_item_matrix)
    # Get similarity matrix
    user_similarity = cosine_similarity(train)
    item_similarity = cosine_similarity(train, 'item')
    # Predict rating
    prediction_user = predict_rating(train, user_similarity)
    prediction_item = predict_rating(train, item_similarity, 'item')

    # Evaulate goodness
    mse_user = evaluation(prediction_user, test)
    mse_item = evaluation(prediction_item, test)
    print('User based CF MSE:'+ str(mse_user))
    print('Item based CF MSE:' + str(mse_item))


    # Validate recommendations
    print(np.argsort(user_similarity[7, :])[:-6-1:-1])


if __name__ == '__main__':
    main()