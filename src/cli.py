import click
import pandas as pd
from .train import train_model
from .package_results import package_experiment_results
from .fairness import evaluate_fairness

@click.group()
def cli():
    # I'm using Click to organize my commands.
    # It makes the CLI nicer to use than sys.argv, in my opinion.
    pass

@cli.command()
@click.argument("data_path")
@click.argument("output_dir")
def run_experiment(data_path, output_dir):
    """
    This command trains a model using data from data_path, 
    then saves the results (model, metrics, etc.) into output_dir.
    """
    # Train the model
    model, metrics, y_true, y_pred = train_model(data_path)

    # I made up this config dictionary.
    config = {
        "model": "RandomForestClassifier"
    }

    # Package everything so we can look at it later
    package_experiment_results("wine_model", model, metrics, config, output_dir)

@cli.command()
@click.argument("data_path")
@click.argument("preds_path")
def check_fairness(data_path, preds_path):
    """
    This command checks how fair our predictions are by comparing them
    to the 'alcohol' feature in the original dataset.
    """
    # Read the original data
    df = pd.read_csv(data_path)
    # Read the list of predictions
    preds = pd.read_csv(preds_path)["predictions"].tolist()

    # Evaluate fairness. Right now it's just a simple function,
    # but ideally I'd add more checks.
    fairness_result = evaluate_fairness(preds, df["alcohol"])
    print(f"Fairness results: {fairness_result}")

if __name__ == "__main__":
    # This starts the CLI when we run "python cli.py"
    cli()
