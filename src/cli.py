import typer
from pathlib import Path
from alphaoracle.utils.analyze_contacts import AvgModelsPDB
from alphaoracle.training.train_embeddings import EmbeddingsTrainer
from alphaoracle.training.train_alphaoracle import AlphaOracleTrainer
from alphaoracle.run_alphaoracle import Predictor

app = typer.Typer(help="AlphaOracle: A tool for protein-protein interaction prediction")


@app.command("avgmodels")
def analyze_avgmodels(config_path: Path):
    """Obtain avg_n_models feature for protein pairs from PDB structures

    Parameters should be specified in a `.json` config file.
    """
    analyzer = AvgModelsPDB()
    analyzer.run_from_config(config_path)
    typer.echo("All structures have been processed.")


@app.command("embeddings")
def train_embeddings(config_path: Path):
    """Trains and generates protein embeddings.

    Parameters should be specified in a `.json` config file.
    """
    trainer = EmbeddingsTrainer(config_path)
    trainer.train()
    trainer.forward()
    typer.echo("Embeddings generated.")


@app.command("train")
def train_classifier(config_path: Path):
    """Trains the AlphaORACLE classifier for protein-protein interaction prediction.

    Parameters should be specified in a `.json` config file.
    """
    trainer = AlphaOracleTrainer(config_path)
    trainer.train()
    typer.echo("AlphaORACLE classifier training complete.")


@app.command("predict")
def predict_interactions(config_path: Path):
    """Predicts protein-protein interactions using the trained AlphaORACLE model.

    Parameters should be specified in a `.json` config file.
    """
    predictor = Predictor(config_path)
    predictor.run()
    typer.echo("PPI predictions complete.")


def main():
    app()