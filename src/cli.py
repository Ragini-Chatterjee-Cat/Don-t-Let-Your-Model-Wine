### src/cli.py
import click
from .train import train_model
from .package_results import package_experiment_results
from .fairness import evaluate_fairness
import pandas as pd

@click.group()
def cli():
    pass

@cli.command()
@click.argument('data_path')
@click.argument('output_dir')
def run_experiment(data_path, output_dir):
    model, metrics, y_true, y_pred = train_model(data_path)
    config = {"model": "RandomForestClassifier"}
    package_experiment_results("wine_model", model, metrics, config, output_dir)

@cli.command()
@click.argument('data_path')
@click.argument('preds_path')
def check_fairness(data_path, preds_path):
    df = pd.read_csv(data_path)
    preds = pd.read_csv(preds_path)['predictions'].tolist()
    fairness = evaluate_fairness(preds, df['alcohol'])
    print(f"Fairness results: {fairness}")

if __name__ == '__main__':
    cli()



