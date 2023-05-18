import numpy as np
import matplotlib.pyplot as plt
import pylab as p
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score, mean_squared_error

model_names = ['ridge', 'knn', 'rfr', 'grad', 'svr', 'krr', 'gpr']

models = {
    'ridge': Ridge(),
    'knn': KNeighborsRegressor(),
    'rfr': RandomForestRegressor(),
    'grad': GradientBoostingRegressor(),
    'svr': SVR(),
    'krr': KernelRidge(),
    'gpr': GaussianProcessRegressor()
}


def score_model(model_name, y_tr_pred, y_tr, y_pred, y_te):
    """
    :param model_name: Already fitted sklearn model
    :param y_tr_pred:  Predicted y from training data
    :param y_tr:  y training data
    :param y_pred:  Predicted y from test data
    :param y_te:  y test data
    :return:  scores: dict ['r2_tr', 'r2_te', 'mse_tr', 'mse_te', 'rmse_tr', rmse_te']
    """
    train_r2 = r2_score(y_tr, y_tr_pred)
    train_mse = mean_squared_error(y_tr, y_tr_pred)
    train_rmse = np.sqrt(train_mse)
    test_r2 = r2_score(y_te, y_pred)
    test_mse = mean_squared_error(y_te, y_pred)
    test_rmse = np.sqrt(test_mse)
    scores = {
        'r2_tr': train_r2,
        'r2_te': test_r2,
        'mse_tr': train_mse,
        'mse_te': test_mse,
        'rmse_tr': train_rmse,
        'rmse_te': test_rmse
    }
    scores = list(scores.values())
    return scores


def classical_regressor(model: str, X_tr, y_tr, X_te, y_te, plot=True, save=False):
    """
    :param model: defines the model to run, raises a ValueError if unexpected type
    :param X_tr: X training data
    :param y_tr: y training data
    :param X_te: X testing data
    :param y_te: y testing data
    :param plot: whether to produce a plot, plot will be predicted vs actual
    :param save: whether to save the plot saves to /plots/model.png
    :return: returns current_scores dict of test scores from score_model
    """
    if model not in model_names:
        raise ValueError('Model must be one of', model_names)
    current_model = models[model]
    current_model.fit(X_tr, y_tr)
    y_tr_pred = current_model.predict(X_tr)
    y_pred = current_model.predict(X_te)
    current_scores = score_model(model, y_tr_pred, y_tr, y_pred, y_te)

    if plot:
        plt.scatter(y_te, y_pred, color='r', label='Test data')
        plt.scatter(y_tr, y_tr_pred, color='b', label='Train data')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        if save:
            plt.savefig('plots/' + model + '.png')
        plt.show()
    return current_scores


def run_models(X_tr, y_tr, X_te, y_te):
    scores = dict()
    keys = ["Train", "Test"]
    for model in model_names:
        scores[model] = classical_regressor(model, X_tr, y_tr, X_te, y_te, plot=False)

    for i in range(3):  # 3 because we have 3 different scoring types, this will loop over each of them and create a
        # bar plot comparing the models for each scoring metric
        heights = list(scores.values())
        heights = np.array(heights).transpose()
        heights = dict(zip(keys, heights[i*2:2*i+2]))  # because keys only has two elements this zips the first two lists in
        # heights which corresponds to r2score train and test respectively.

        width = 0.25
        multiplier = 0
        bar_clusters = np.arange(len(scores.keys()))

        fig, ax = plt.subplots(layout='constrained')

        for key, height in heights.items():
            offset = width * multiplier
            bar_plt = plt.bar(bar_clusters + offset, height, width, label=key)
            plt.bar_label(bar_plt)
            multiplier += 1
        ax.set_xticks(bar_clusters + width, scores.keys())
        ax.legend(loc='upper left', ncols=2)
        titles = ['R2 Scores', 'MSE', 'RMSE']
        ax.set_title(titles[i])
        plt.show()
