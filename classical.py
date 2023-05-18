import numpy as np
import matplotlib.pyplot as plt
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
        model_name + '_r2_tr': train_r2,
        model_name + '_r2_te': test_r2,
        model_name + '_mse_tr': train_mse,
        model_name + '_mse_te': test_mse,
        model_name + '_rmse_tr': train_rmse,
        model_name + '_rmse_te': test_rmse
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
            plt.savefig('plots/'+model+'.png')
        plt.show()
    return current_scores


def run_models(X_tr, y_tr, X_te, y_te):
    score = []
    for i in range(len(model_names)):
        score.append(classical_regressor(model_names[i], X_tr, y_tr, X_te, y_te, plot=False))
    fig = plt.figure()
    plot_spacing = np.arange(len(score[0]))
    ax = fig.add_axes([0, 0, 1, 1])
    print(score)
    print(np.shape(score))
    for count in score:
        print(np.shape(count))

    for i in range(len(model_names)):
        ax.bar(plot_spacing + i / len(model_names), score[i], width=.2)
    plt.show()
