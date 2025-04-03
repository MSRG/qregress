import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from .quantum.Evaluate import evaluate

gaussian_kernel = RBF()

models = {
    'ridge': Ridge(),
    'lasso': Lasso(),
    'elastic': ElasticNet(),
    'knn': KNeighborsRegressor(),
    'rfr': RandomForestRegressor(),
    'grad': GradientBoostingRegressor(),
    'svr': SVR(),
    'krr': KernelRidge(),
    'gpr': GaussianProcessRegressor()
}

param_grid = {
    'ridge': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100, 1000]
    },
    'knn': {
        'n_neighbors': range(1, 10),
        'weights': ['uniform', 'distance']
    },
    'rfr': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    'grad': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.01, 0.001],
        'max_depth': [3, 5, 7]
    },
    'svr': {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.01, 0.001]
    },
    'krr': {
        'kernel': ['linear', 'poly', 'rbf'],
        'alpha': [0.001, 0.01, 0.1, 1],
        'gamma': [1, 0.1, 0.01, 0.001]
    },
    'gpr': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        'kernel': [gaussian_kernel]
    },
    'lasso': {
        'alpha': [0.001, 0.01, 0.1, 1],
        'max_iter': [1000, 2000, 3000]
    },
    'elastic': {
        'alpha': [0.001, 0.01, 0.1, 1],
        'l1_ratio': [0.2, 0.5, 0.8],
        'max_iter': [1000, 2000, 3000]
    }
}

'''
def score_model(model, sacler, X_tr, y_tr, X_te, y_te):
    """
    :param X_tr:
    :param model:
    :param X_te:
    :param y_tr:  y training data
    :param y_te:  y test data
    :return:  scores: dict ['r2_tr', 'r2_te', 'mse_tr', 'mse_te', 'rmse_tr', rmse_te']
    """
    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)
    train_r2 = r2_score(y_tr, y_tr_pred)
    train_mse = mean_squared_error(y_tr, y_tr_pred)
    train_mae = mean_absolute_error(y_tr, y_tr_pred)
    test_r2 = r2_score(y_te, y_te_pred)
    test_mse = mean_squared_error(y_te, y_te_pred)
    test_mae = mean_absolute_error(y_te, y_te_pred)
    scores = {
        'r2_tr': train_r2,
        'r2_te': test_r2,
        'mse_tr': train_mse,
        'mse_te': test_mse,
        'mae_tr': train_mae,
        'mae_te': test_mae
    }
    print(scores['mae_tr'], scores['mae_te'])
    scores = list(scores.values())
    return scores
'''


def classical_regressor(model: str, scaler, X_tr, y_tr, X_te, y_te, plot=True, save=False):
    """
    :param scaler:
    :param model: defines the model to run, raises a ValueError if unexpected type
    :param X_tr: X training data
    :param y_tr: y training data
    :param X_te: X testing data
    :param y_te: y testing data
    :param plot: whether to produce a plot, plot will be predicted vs actual
    :param save: whether to save the plot saves to /plots/model.png
    :return: returns current_scores dict of test scores from evaluate function
    """
    if model not in models.keys():
        raise ValueError('Model must be one of', models.keys())

    st = time.time()
    current_model = models[model]
    current_param_grid = param_grid[model]

    grid_search = GridSearchCV(current_model, current_param_grid, cv=5)

    print(f'Now fitting {model}... ')

    grid_search.fit(X_tr, y_tr)

    best_model = grid_search.best_estimator_

    print(f'Completed fitting {model} in {time.time() - st} seconds. ')

    current_scores, y_te_pred, y_tr_pred = evaluate(model=best_model, X_train=X_tr, X_test=X_te, y_train=y_tr,
                                                    y_test=y_te, y_scaler=scaler, plot=True, title=model)
    if plot:
        plt.scatter(y_te, y_te_pred, color='r', label='Test data')
        plt.scatter(y_tr, y_tr_pred, color='b', label='Train data')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        if save:
            plt.savefig('plots/' + model + '.png')
        plt.show()
    return current_scores, y_te_pred, y_tr_pred


def run_models(scaler, X_tr, y_tr, X_te, y_te, save_plots):
    scores = dict()
    keys = ["Train", "Test"]
    y_te_pred = {}
    y_tr_pred = {}
    for model in models.keys():
        scores[model], y_te_pred[model], y_tr_pred[model] = classical_regressor(model, scaler, X_tr, y_tr, X_te, y_te,
                                                                                plot=save_plots)

    """
    for i in range(3):  # 3 because we have 3 different scoring types, this will loop over each of them and create a
        # bar plot comparing the models for each scoring metric
        heights = list(scores.values())
        heights = np.array(heights).transpose()
        heights = dict(zip(keys, heights[i*2:2*i+2]))  # because keys only has two elements this zips the first two
        # lists in ith set of lists which corresponds to a score type.

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
        titles = ['R2 Scores', 'MSE', 'MAE']
        ax.set_title(titles[i])
        plt.show()
        if save_plots:
            plt.savefig('plots/model_comparison.svg')

    """

    return scores, y_tr_pred, y_te_pred
