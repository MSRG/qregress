from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_1d(model, X_train, X_test, y_train, y_test, plot=True, title=""):
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


def evaluate(model, X_train, y_train, X_test=None, y_test=None, plot: bool = False, title: str = 'defult'):
    scores = {}

    y_train_pred = model.predict(X_train)
    scores['MSE_train'] = mean_squared_error(y_train, y_train_pred),
    scores['R2_train'] = r2_score(y_train, y_train_pred)

    y_test_pred = None
    if y_test is not None:
        y_test_pred = model.predict(X_test)
        scores['MSE_test'] = mean_squared_error(y_test, y_test_pred)
        scores['R2_test'] = r2_score(y_test, y_test_pred)

    if plot:
        if y_test_pred is not None:
            plt.scatter(y_test, y_test_pred, color='b', s=10, label='Test')
        plt.scatter(y_train, y_train_pred, color='r', s=10, label='Train')
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.legend()
        plt.savefig(title+'_plot.svg')
    return scores
