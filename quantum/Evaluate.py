import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_train, X_test, y_train, y_test, plot=True, title=""):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    scores = {'MSE_train': mean_squared_error(y_train, y_train_pred),
              'MSE_test': mean_squared_error(y_test, y_test_pred),
              'R2_train': r2_score(y_train, y_train_pred),
              'R2_test': r2_score(y_test, y_test_pred)
              }
    if plot:
        plt.title(title)
        plt.scatter(X_train, y_train_pred, color='b', label='Train', s=10)
        plt.scatter(X_test, y_test_pred, color='orange', label='Test', s=10)
        plt.scatter(X_train, y_train, color='green', label='Data', s=10)
        plt.scatter(X_test, y_test, color='green', s=10)
        plt.legend()
    return scores


def grid_searh(model, x_train, x_test, y_train, y_test, hyperparameters: dict, **kwargs):
    """
    Performs a grid search on the given model. Trains the model for each combination of hyperparameters, and then
    trains it using x_train, y_train. Scores each model using r2_score on the test dataset and returns the best
    performing model with its score and hyperparameters. Any additional parameters to be passed to the model are
    handled with kwargs.

    :return: trained_model, dict: best_hyperparameters, flaot: best_score, list: results
    """
    for x in hyperparameters.values():
        if not isinstance(x, list):
            raise ValueError('Dictionary must contain lists of values to try! ')

    results = []
    best_score = float('-inf')
    best_model = None
    best_hyperparameters = {}

    param_combinations = list(itertools.product(*hyperparameters.values()))
    for combination in param_combinations:
        update = dict(zip(hyperparameters.keys(), combination))
        kwargs.update(update)

        built_model = model(**kwargs)
        built_model.fit(x_train, y_train, callback_interval=1)
        y_pred = built_model.predict(x_test)
        score = r2_score(y_test, y_pred)
        results.append(score)

        if score > best_score:
            best_score = score
            best_model = built_model
            best_hyperparameters = {key: kwargs[key] for key in hyperparameters.keys()}

    return best_model, best_hyperparameters, best_score, results
