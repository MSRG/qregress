import pandas as pd
import click
@click.command()
@click.option('--path', required=True, type=click.Path(exists=True), help='Path to model_log.csv')
def main(path):
    df = pd.read_csv(path).dropna()
    print(df.shape[0])

if __name__ == "__main__":
    main()
