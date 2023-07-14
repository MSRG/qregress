import click
import joblib
import json
import pandas as pd
import numpy as np

from Classical import run_models


@click.command()
@click.option('--scaler', type=click.Path(exists=True), required=True, help='File for y scaler to unscale after '
                                                                            'prediction')
@click.option('--train_set', type=click.Path(exists=True), required=True, help='File for train set')
@click.option('--test_set', type=click.Path(exists=True), required=True, help='File for test set')
@click.option('--save_plots', default=False, help="Don't use: depreceating soon... ")
def main(scaler, train_set, test_set, save_plots):
    train = joblib.load(train_set)
    test = joblib.load(test_set)
    X_tr = train['X']
    X_te = test['X']
    y_tr = np.array(train['y']).reshape(-1)
    y_te = np.array(test['y']).reshape(-1)
    scaler = joblib.load(scaler)

    scores, y_tr_pred, y_te_pred = run_models(scaler, X_tr, y_tr, X_te, y_te, save_plots)

    y_tr = scaler.inverse_transform(y_tr.reshape(-1, 1))
    y_te = scaler.inverse_transform(y_te.reshape(-1, 1))

    for (model_name, train_pred), (test_pred) in zip(y_tr_pred.items(), y_te_pred.values()):
        name = model_name + '_predicted_values.csv'
        train_pred = np.array(train_pred).reshape(-1).tolist()
        y_tr = np.array(y_tr).reshape(-1).tolist()
        test_pred = np.array(test_pred).reshape(-1).tolist()
        y_te = np.array(y_te).reshape(-1).tolist()

        df_train = pd.DataFrame({'Predicted': train_pred, 'Reference': y_tr})
        df_train['Data'] = 'Train'
        df_test = pd.DataFrame({'Predicted': test_pred, 'Reference': y_te})
        df_test['Data'] = 'Test'
        df = pd.concat([df_train, df_test], ignore_index=True)
        df = df[['Data', 'Predicted', 'Reference']]

        df.to_csv(name, index=False)
        print(f'Saved predicted values as {name}')

    with open('scores.json', 'w') as outfile:
        json.dump(scores, outfile)
        print(f'Scores saved as {outfile.name}. ')


if __name__ == '__main__':
    main()
