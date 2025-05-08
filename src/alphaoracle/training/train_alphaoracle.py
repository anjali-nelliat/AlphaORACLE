import os
import time
import json
import warnings
from pathlib import Path
from typing import Union, List, Dict, Tuple, Any, Optional

import typer
import numpy as np
import pandas as pd
import optuna
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, confusion_matrix, roc_curve, matthews_corrcoef,
                             classification_report)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


from alphaoracle.utils.classifier_args import ClassifierArgsParser
from alphaoracle.utils.data_loader import load_ortholog_mapping, load_embeddings, load_af_scores, load_avg_n_models, create_edge_embedding


class ProteinInteractionDataset(Dataset):
    def __init__(self, interaction_data, human_embeddings, yeast_embeddings,
                 human_to_yeast, af_scores, avg_n_models, human_dim=512, yeast_dim=512):
        self.interaction_data = interaction_data
        self.human_embeddings = human_embeddings
        self.yeast_embeddings = yeast_embeddings
        self.human_to_yeast = human_to_yeast
        self.af_scores = af_scores
        self.avg_n_models = avg_n_models
        self.human_dim = human_dim
        self.yeast_dim = yeast_dim

    def __len__(self):
        return len(self.interaction_data)

    def __getitem__(self, idx):
        # Convert the label to float first
        human_protein1, human_protein2, label = self.interaction_data[idx]
        try:
            label = float(label)  # Ensure label is a float
        except (ValueError, TypeError):
            print(f"Warning: Invalid label value: {label}, defaulting to 0.0")
            label = 0.0

        # Get yeast orthologs
        yeast_protein1 = self.human_to_yeast.get(human_protein1)
        yeast_protein2 = self.human_to_yeast.get(human_protein2)

        # Get human embeddings
        human_emb1 = self.human_embeddings.get(human_protein1, np.zeros(self.human_dim))
        human_emb2 = self.human_embeddings.get(human_protein2, np.zeros(self.human_dim))

        # Get yeast embeddings (if orthologs exist)
        yeast_emb1 = self.yeast_embeddings.get(yeast_protein1,
                                               np.zeros(self.yeast_dim)) if yeast_protein1 else np.zeros(self.yeast_dim)
        yeast_emb2 = self.yeast_embeddings.get(yeast_protein2,
                                               np.zeros(self.yeast_dim)) if yeast_protein2 else np.zeros(self.yeast_dim)

        # Create edge embeddings using cosine similarity
        human_similarity = create_edge_embedding(human_emb1, human_emb2)
        yeast_similarity = create_edge_embedding(yeast_emb1, yeast_emb2)

        # Combine similarities
        combined_features = np.concatenate([human_similarity, yeast_similarity])

        # Get AlphaFold confidence score
        af_score = self.af_scores.get((human_protein1, human_protein2), 0.0)
        # Ensure AF score is a float
        try:
            af_score = float(af_score)
        except (ValueError, TypeError):
            print(f"Warning: Invalid AF score: {af_score}, defaulting to 0.0")
            af_score = 0.0

        # Get avg_n_models score
        avg_n_models_score = self.avg_n_models.get((human_protein1, human_protein2), 0.0)
        # Ensure avg_n_models score is a float
        try:
            avg_n_models_score = float(avg_n_models_score)
        except (ValueError, TypeError):
            print(f"Warning: Invalid avg_n_models score: {avg_n_models_score}, defaulting to 0.0")
            avg_n_models_score = 0.0

        # Final feature vector is combined similarities + AF score + avg_n_models score
        feature_vector = np.append(combined_features, [af_score, avg_n_models_score])

        # Ensure all values are floats
        feature_vector = feature_vector.astype(np.float32)

        return torch.FloatTensor(feature_vector), torch.FloatTensor([label])


class InteractionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rates, activation_function=nn.ReLU):
        super(InteractionClassifier, self).__init__()

        # Build network layers dynamically based on hidden_dims list
        layers = []
        prev_dim = input_dim

        for i, (dim, dropout) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation_function())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AlphaOracleTrainer:
    def __init__(self, config: Union[Path, Dict]):
        """Initialize the AlphaOracle trainer with configuration.

        Args:
            config: Path to config file or dictionary containing configuration parameters
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        typer.secho(f"Using {self.device}", fg=("green" if self.device.type == "cuda" else "yellow"))

        # Parse configuration
        self.params = self.parse_config(config)

        # Setup output paths
        self.setup_output_paths()

        # Initialize data containers
        self.yeast_to_human = None
        self.human_to_yeast = None
        self.human_embeddings = None
        self.yeast_embeddings = None
        self.af_scores = None
        self.avg_n_models = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.input_dim = None

        # Training results
        self.best_model = None
        self.best_params = None
        self.study = None
        self.test_preds = None
        self.test_labels = None
        self.test_protein_pairs = None

    def parse_config(self, config: Union[Path, Dict]) -> Any:
        """Parse the configuration file or dictionary.

        Args:
            config: Path to config file or dictionary with parameters

        Returns:
            Parsed configuration namespace
        """
        parser = ClassifierArgsParser(config)
        return parser.parse()

    def setup_output_paths(self) -> None:
        """Setup output directory and file paths."""
        if self.params.output_dir:
            os.makedirs(self.params.output_dir, exist_ok=True)
            self.model_path = os.path.join(self.params.output_dir, self.params.model_file)
            self.study_path = os.path.join(self.params.output_dir, self.params.study_file)
            self.predictions_path = os.path.join(self.params.output_dir, self.params.predictions_file)
            self.performance_plot_path = os.path.join(self.params.output_dir, self.params.performance_plot)
            self.optim_history_path = os.path.join(self.params.output_dir, self.params.optimization_history_plot)
            self.param_import_path = os.path.join(self.params.output_dir, self.params.param_importances_plot)
            self.train_history_path = os.path.join(self.params.output_dir, self.params.training_history_plot)
        else:
            self.model_path = self.params.model_file
            self.study_path = self.params.study_file
            self.predictions_path = self.params.predictions_file
            self.performance_plot_path = self.params.performance_plot
            self.optim_history_path = self.params.optimization_history_plot
            self.param_import_path = self.params.param_importances_plot
            self.train_history_path = self.params.training_history_plot

    def load_data(self) -> None:
        """Load all necessary data files."""
        typer.echo("Loading data...")

        # Load ortholog mapping
        self.yeast_to_human, self.human_to_yeast = load_ortholog_mapping(self.params.ortholog_mapping_path)

        # Load embeddings
        self.human_embeddings, self.yeast_embeddings, loaded_human_dim, loaded_yeast_dim = load_embeddings(
            self.params.human_embeddings_path,
            self.params.yeast_embeddings_path,
            node_column="node")

        # Use dimensions from data if specified dimensions don't match
        if loaded_human_dim != self.params.human_dim:
            typer.secho(
                f"Warning: Specified human_dim ({self.params.human_dim}) doesn't match data dimension "
                f"({loaded_human_dim}). Using {loaded_human_dim}.",
                fg="yellow"
            )
            self.params.human_dim = loaded_human_dim

        if loaded_yeast_dim != self.params.yeast_dim:
            typer.secho(
                f"Warning: Specified yeast_dim ({self.params.yeast_dim}) doesn't match data dimension "
                f"({loaded_yeast_dim}). Using {loaded_yeast_dim}.",
                fg="yellow"
            )
            self.params.yeast_dim = loaded_yeast_dim

        # Load scores
        self.af_scores = load_af_scores(self.params.af_scores_path)
        self.avg_n_models = load_avg_n_models(self.params.avg_n_models_path)

    def prepare_datasets(self) -> None:
        """Prepare training, validation, and test datasets."""
        typer.echo("Preparing datasets...")

        # Load dataset with interacting pairs
        full_data = pd.read_csv(self.params.training_data_path)

        # Check if the required columns exist
        required_columns = ["Protein1", "Protein2", "Class"]
        for col in required_columns:
            if col not in full_data.columns:
                raise ValueError(
                    f"Column '{col}' not found in training data. Available columns: {full_data.columns.tolist()}")

        # Split data into train, validation, and test sets using config values
        train_val_df, test_df = train_test_split(
            full_data,
            test_size=self.params.train_test_split_ratio,
            random_state=self.params.random_seed,
            stratify=full_data["Class"]
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.params.train_val_split_ratio,
            random_state=self.params.random_seed,
            stratify=train_val_df["Class"]
        )

        typer.echo(f"Full dataset size: {len(full_data)}")
        typer.echo(
            f"Train data size ({100 * (1 - self.params.train_test_split_ratio) * (1 - self.params.train_val_split_ratio):.0f}% of total): {len(train_df)}")
        typer.echo(
            f"Validation data size ({100 * (1 - self.params.train_test_split_ratio) * self.params.train_val_split_ratio:.0f}% of total): {len(val_df)}")
        typer.echo(f"Test data size ({100 * self.params.train_test_split_ratio:.0f}% of total): {len(test_df)}")

        # Print class distribution
        typer.echo("\nClass distribution in each split:")
        typer.echo(f"Train: {train_df['Class'].value_counts().to_dict()}")
        typer.echo(f"Validation: {val_df['Class'].value_counts().to_dict()}")
        typer.echo(f"Test: {test_df['Class'].value_counts().to_dict()}")

        # Convert to list of tuples
        train_data = [(str(row["Protein1"]), str(row["Protein2"]), float(row["Class"])) for _, row in
                      train_df.iterrows()]
        val_data = [(str(row["Protein1"]), str(row["Protein2"]), float(row["Class"])) for _, row in val_df.iterrows()]
        test_data = [(str(row["Protein1"]), str(row["Protein2"]), float(row["Class"])) for _, row in test_df.iterrows()]

        # Store test protein pairs for prediction output
        self.test_protein_pairs = [(protein1, protein2) for protein1, protein2, _ in test_data]

        # Create datasets
        self.train_dataset = ProteinInteractionDataset(
            train_data,
            self.human_embeddings,
            self.yeast_embeddings,
            self.human_to_yeast,
            self.af_scores,
            self.avg_n_models,
            self.params.human_dim,
            self.params.yeast_dim
        )
        self.val_dataset = ProteinInteractionDataset(
            val_data,
            self.human_embeddings,
            self.yeast_embeddings,
            self.human_to_yeast,
            self.af_scores,
            self.avg_n_models,
            self.params.human_dim,
            self.params.yeast_dim
        )
        self.test_dataset = ProteinInteractionDataset(
            test_data,
            self.human_embeddings,
            self.yeast_embeddings,
            self.human_to_yeast,
            self.af_scores,
            self.avg_n_models,
            self.params.human_dim,
            self.params.yeast_dim
        )

        # Determine input dimension
        sample_features, _ = next(iter(DataLoader(self.train_dataset, batch_size=1)))
        self.input_dim = sample_features.shape[1]
        typer.echo(f"Input dimension: {self.input_dim}")

    def _objective(self, trial) -> float:
        """Objective function for Optuna hyperparameter optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation AUC score
        """
        # Define hyperparameters to optimize
        n_layers = trial.suggest_int('n_layers', 1, 3)

        # Define hidden dimensions for each layer
        hidden_dims = []
        dropout_rates = []

        for i in range(n_layers):
            # Use suggest_int for consistent sampling between trials
            hidden_dims.append(trial.suggest_int(f'hidden_dim_{i}', 16, 128))
            dropout_rates.append(trial.suggest_float(f'dropout_{i}', 0.1, 0.5))

        # Learning rate (log scale)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)

        # Batch size
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        # Other potential hyperparameters
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

        # Create data loaders with the suggested batch size
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        # Determine activation function
        activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'ELU'])
        if activation_name == 'LeakyReLU':
            activation = lambda: nn.LeakyReLU(trial.suggest_float('leaky_relu_slope', 0.01, 0.3))
        elif activation_name == 'ELU':
            activation = nn.ELU
        else:
            activation = nn.ReLU

        # Create model
        model = InteractionClassifier(self.input_dim, hidden_dims, dropout_rates, activation_function=activation)

        # Move model to device
        model = model.to(self.device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()

        # Early stopping setup
        best_val_auc = 0
        no_improve_count = 0
        max_epochs = 15  # Fixed max epochs for hyperparameter tuning
        patience = 5  # Fixed patience for hyperparameter tuning

        for epoch in range(max_epochs):
            # Training
            model.train()
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(features)
                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_labels.extend(labels.cpu().numpy().flatten())

            val_auc = roc_auc_score(val_labels, val_preds)

            # Report intermediate metric
            trial.report(val_auc, epoch)

            # Early stopping check
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve_count = 0
            else:
                no_improve_count += 1

            # If no improvement for 'patience' epochs, stop training
            if no_improve_count >= patience:
                break

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_auc

    def optimize_hyperparameters(self) -> None:
        """Run hyperparameter optimization using Optuna."""
        typer.echo("Starting Bayesian optimization for hyperparameter tuning...")

        # Create an Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name="protein_interaction_mlp",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        # Optimize with fixed parameters for the objective function
        study.optimize(
            self._objective,
            n_trials=self.params.n_trials,
            timeout=self.params.timeout
        )

        # Print study statistics
        typer.echo("Number of finished trials: " + str(len(study.trials)))
        typer.echo("\nBest trial:")
        best_trial = study.best_trial
        typer.echo("  Value: " + str(best_trial.value))
        typer.echo("  Params: ")
        for key, value in best_trial.params.items():
            typer.echo(f"    {key}: {value}")

        # Create a model with the best parameters
        n_layers = best_trial.params['n_layers']
        hidden_dims = [best_trial.params[f'hidden_dim_{i}'] for i in range(n_layers)]
        dropout_rates = [best_trial.params[f'dropout_{i}'] for i in range(n_layers)]

        # Determine activation function from best trial
        activation_name = best_trial.params['activation']
        if activation_name == 'LeakyReLU':
            activation = lambda: nn.LeakyReLU(best_trial.params['leaky_relu_slope'])
        elif activation_name == 'ELU':
            activation = nn.ELU
        else:
            activation = nn.ReLU

        # Create the optimized model
        best_model = InteractionClassifier(self.input_dim, hidden_dims, dropout_rates, activation_function=activation)

        # Save optimization plots
        self._plot_optimization_history(study)
        self._plot_param_importances(study)

        # Store results
        self.best_model = best_model
        self.best_params = best_trial.params
        self.study = study

    def _plot_optimization_history(self, study) -> None:
        """Plot optimization history from Optuna study.

        Args:
            study: Completed Optuna study
        """
        try:
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(self.optim_history_path)
            plt.close()
            typer.echo(f"Saved optimization history plot to '{self.optim_history_path}'")
        except Exception as e:
            typer.secho(f"Warning: Could not create optimization history plot: {e}", fg="yellow")

    def _plot_param_importances(self, study) -> None:
        """Plot parameter importances from Optuna study.

        Args:
            study: Completed Optuna study
        """
        try:
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(self.param_import_path)
            plt.close()
            typer.echo(f"Saved parameter importances plot to '{self.param_import_path}'")
        except Exception as e:
            typer.secho(f"Warning: Could not create parameter importances plot: {e}", fg="yellow")

    def train_model(self) -> None:
        """Train the model with the best hyperparameters."""
        typer.echo("\nTraining final model with best hyperparameters...")

        # Create data loaders with best batch size
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.best_params['batch_size'], shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.best_params['batch_size'])
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.best_params['batch_size'])

        # Move model to device
        self.best_model = self.best_model.to(self.device)

        # Optimizer and criterion
        optimizer = optim.Adam(self.best_model.parameters(), lr=self.best_params['lr'])
        criterion = nn.BCELoss()

        # Training setup
        best_val_auc = 0
        best_model_state = None
        no_improve_count = 0

        # Lists to track metrics
        train_losses = []
        val_aucs = []
        val_aps = []
        test_aucs = []
        test_aps = []

        # Start training
        for epoch in range(self.params.epochs):
            # Training
            self.best_model.train()
            train_loss = 0

            # Progress bar for training batches
            with typer.progressbar(self.train_loader, label=f"Epoch {epoch + 1}/{self.params.epochs}") as progress:
                for features, labels in progress:
                    # Move data to device
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.best_model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            self.best_model.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for features, labels in self.val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.best_model(features)
                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_labels.extend(labels.cpu().numpy().flatten())

            val_auc = roc_auc_score(val_labels, val_preds)
            val_ap = average_precision_score(val_labels, val_preds)
            val_aucs.append(val_auc)
            val_aps.append(val_ap)

            # Test evaluation
            test_preds = []
            test_labels = []

            with torch.no_grad():
                for features, labels in self.test_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.best_model(features)
                    test_preds.extend(outputs.cpu().numpy().flatten())
                    test_labels.extend(labels.cpu().numpy().flatten())

            test_auc = roc_auc_score(test_labels, test_preds)
            test_ap = average_precision_score(test_labels, test_preds)
            test_aucs.append(test_auc)
            test_aps.append(test_ap)

            # Print metrics
            typer.echo(
                f"Loss: {avg_train_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, "
                f"Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}"
            )

            # Early stopping check
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.best_model.state_dict().copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # If no improvement for 'patience' epochs, stop training
            if no_improve_count >= self.params.patience:
                typer.echo(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation AUC")
                break

        # Load best model
        if best_model_state:
            self.best_model.load_state_dict(best_model_state)

        # Plot training history
        plt.figure(figsize=(15, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        # Plot validation metrics
        plt.subplot(1, 2, 2)
        plt.plot(val_aucs, label='Validation AUC')
        plt.plot(val_aps, label='Validation AP')
        plt.plot(test_aucs, label='Test AUC')
        plt.plot(test_aps, label='Test AP')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation and Test Metrics')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.train_history_path)
        plt.close()

        typer.echo(f"Training history saved to '{self.train_history_path}'")

    def evaluate(self) -> None:
        """Evaluate the trained model on the test set."""
        typer.echo("\nEvaluating final model performance...")

        # Get predictions on test set
        self.best_model.eval()
        self.test_preds = []
        self.test_labels = []

        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.best_model(features)
                self.test_preds.extend(outputs.cpu().numpy().flatten())
                self.test_labels.extend(labels.cpu().numpy().flatten())

        # Save predictions
        predictions_df = pd.DataFrame({
            'protein1': [pair[0] for pair in self.test_protein_pairs],
            'protein2': [pair[1] for pair in self.test_protein_pairs],
            'True_Label': self.test_labels,
            'Predicted_Probability': self.test_preds
        })

        predictions_df.to_csv(self.predictions_path, index=False)
        typer.echo(f"Saved test predictions to {self.predictions_path}")

        # Run full evaluation
        self._evaluate_model_performance(self.test_labels, self.test_preds)

    def _evaluate_model_performance(self, true_labels, predicted_probs, threshold=0.5) -> Dict[str, float]:
        """Evaluate model performance with various metrics and plots.

        Args:
            true_labels: Ground truth binary labels
            predicted_probs: Predicted probabilities from the model
            threshold: Threshold for binary classification

        Returns:
            Dictionary with performance metrics
        """
        # Convert probabilities to binary predictions using threshold
        predicted_labels = (np.array(predicted_probs) >= threshold).astype(int)
        true_labels = np.array(true_labels)

        # Find optimal F1 threshold and calculate metrics
        f1_scores = []
        thresholds_to_test = np.linspace(0.1, 0.9, 9)
        for thresh in thresholds_to_test:
            pred_labels = (np.array(predicted_probs) >= thresh).astype(int)
            f1_scores.append(f1_score(true_labels, pred_labels))

        # Find optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_to_test[optimal_idx]

        # Use the optimal threshold for all metrics
        optimal_predictions = (np.array(predicted_probs) >= optimal_threshold).astype(int)
        accuracy = accuracy_score(true_labels, optimal_predictions)
        conf_matrix = confusion_matrix(true_labels, optimal_predictions)
        f1 = f1_score(true_labels, optimal_predictions)
        mcc = matthews_corrcoef(true_labels, optimal_predictions)

        # Get detailed classification report
        report = classification_report(true_labels, predicted_labels)

        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(true_labels, predicted_probs)
        roc_auc = roc_auc_score(true_labels, predicted_probs)

        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(true_labels, predicted_probs)
        avg_precision = average_precision_score(true_labels, predicted_probs)

        # We already calculated these metrics above
        optimal_f1 = f1_scores[optimal_idx]

        # Print metrics
        typer.echo("\n============ Model Performance Metrics ============")
        typer.echo(f"Accuracy: {accuracy:.4f}")
        typer.echo(f"F1 Score (threshold={optimal_threshold:.2f}): {f1:.4f}")
        typer.echo(f"Matthews Correlation Coefficient: {mcc:.4f}")
        typer.echo(f"ROC AUC: {roc_auc:.4f}")
        typer.echo(f"Average Precision: {avg_precision:.4f}")
        typer.echo(f"Optimal F1 Score: {optimal_f1:.4f} at threshold: {optimal_threshold:.2f}")
        typer.echo("\nConfusion Matrix:")
        typer.echo(str(conf_matrix))
        typer.echo("\nClassification Report:")
        typer.echo(report)

        # Create visualizations
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(2, 3, figure=fig)

        # ROC curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'r--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc='lower right')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Precision-Recall curve
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(recall, precision, 'g-', label=f'PR (AP = {avg_precision:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc='lower left')
        ax2.grid(True, linestyle='--', alpha=0.6)

        # F1 score vs threshold
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(thresholds_to_test, f1_scores, 'o-', color='purple')
        ax3.axvline(x=optimal_threshold, color='r', linestyle='--',
                    label=f'Optimal threshold: {optimal_threshold:.2f}')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score vs Threshold')
        ax3.legend(loc='lower center')
        ax3.grid(True, linestyle='--', alpha=0.6)

        # Calculate confusion matrix using optimal F1 threshold
        optimal_predictions = (np.array(predicted_probs) >= optimal_threshold).astype(int)
        optimal_conf_matrix = confusion_matrix(true_labels, optimal_predictions)

        # Confusion Matrix heatmap with optimal threshold
        ax4 = fig.add_subplot(gs[1, 0])
        sns.heatmap(optimal_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Interaction', 'Interaction'],
                    yticklabels=['No Interaction', 'Interaction'])
        ax4.set_xlabel('Predicted Label')
        ax4.set_ylabel('True Label')
        ax4.set_title(f'Confusion Matrix (threshold={optimal_threshold:.2f})')

        # Probability distribution
        ax5 = fig.add_subplot(gs[1, 1:])

        # Separate positive and negative examples
        pos_probs = [predicted_probs[i] for i in range(len(predicted_probs)) if true_labels[i] == 1]
        neg_probs = [predicted_probs[i] for i in range(len(predicted_probs)) if true_labels[i] == 0]

        ax5.hist(pos_probs, bins=20, alpha=0.5, color='green', label='Positive Examples')
        ax5.hist(neg_probs, bins=20, alpha=0.5, color='red', label='Negative Examples')
        ax5.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold: {threshold}')
        ax5.axvline(x=optimal_threshold, color='purple', linestyle='--',
                    label=f'Optimal F1 Threshold: {optimal_threshold:.2f}')
        ax5.set_xlabel('Predicted Probability')
        ax5.set_ylabel('Count')
        ax5.set_title('Distribution of Predicted Probabilities')
        ax5.legend()
        ax5.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.performance_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        typer.echo(f"\nPerformance visualization saved as '{self.performance_plot_path}'")

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'mcc': mcc,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_f1
        }

    def save_model(self) -> None:
        """Save the trained model and study."""
        # Save the model
        torch.save(self.best_model.state_dict(), self.model_path)
        typer.echo(f"Saved optimized model to '{self.model_path}'")

        # Save the hyperparameter optimization results
        try:
            joblib.dump(self.study, self.study_path)
            typer.echo(f"Saved Optuna study to '{self.study_path}'")
        except Exception as e:
            typer.secho(f"Warning: Could not save Optuna study: {e}", fg="yellow")

    def get_output_paths(self) -> Dict[str, str]:
        """Get the paths of all output files."""
        return {
            'model_path': self.model_path,
            'study_path': self.study_path,
            'predictions_path': self.predictions_path,
            'performance_plot_path': self.performance_plot_path,
            'optimization_history_path': self.optim_history_path,
            'param_importances_path': self.param_import_path,
            'training_history_path': self.train_history_path
        }

    def train(self, verbosity: int = 1) -> Dict[str, str]:
        """Run the full training pipeline.

        Args:
            verbosity: 0 to suppress printing (except progress bars), 1 for regular printing

        Returns:
            Dictionary with paths to output files
        """
        # 1. Load data
        self.load_data()

        # 2. Prepare datasets
        self.prepare_datasets()

        # 3. Run hyperparameter optimization
        self.optimize_hyperparameters()

        # 4. Train the model with best hyperparameters
        self.train_model()

        # 5. Save the model
        self.save_model()

        # 6. Evaluate the model
        self.evaluate()

        # 7. Return output paths
        typer.echo("\nTraining completed successfully!")
        return self.get_output_paths()