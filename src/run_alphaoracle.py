import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import json
import time
from pathlib import Path
from tqdm.notebook import tqdm
from typing import Dict, Any, Optional, Union, List, Tuple
from scipy.spatial.distance import cosine

from alphaoracle.utils.data_loader import load_ortholog_mapping, load_embeddings, load_af_scores, load_avg_n_models, create_edge_embedding
from alphaoracle.utils.classifier_args import PredictionArgsParser


# MLP Classifier with probability output
class InteractionClassifier(nn.Module):
    def __init__(self):
        super(InteractionClassifier, self).__init__()

        input_dim = 4
        n_layers = 2
        hidden_dims = [30, 116]
        dropout_rates = [0.153280441, 0.103606574]
        activation_function = nn.LeakyReLU(0.227876086)

        # Build network layers dynamically
        layers = []
        prev_dim = input_dim

        for i, (dim, dropout) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation_function)
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Predictor:
    """A class to predict protein-protein interactions using a trained model."""

    def __init__(self, config):
        """Initialize the predictor with configuration.

        Args:
            config: Path to a JSON configuration file or a dictionary of parameters
        """
        # Process configuration
        parser = PredictionArgsParser(config)
        self.params = parser.parse()

        # Create output directory
        self.output_dir = Path(self.params["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        if self.params["device"] is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.params["device"])

        # Initialize model and data attributes to None
        self.model = None
        self.human_embeddings = None
        self.yeast_embeddings = None
        self.human_to_yeast = None
        self.yeast_to_human = None
        self.af_scores = None
        self.avg_n_models = None

        # Load resources
        self._load_resources()

    def _load_resources(self):
        """Load all necessary resources for prediction."""
        self._load_model()
        self._load_embeddings()
        self._load_orthologs()
        self._load_scores()

    def _load_model(self):
        """Load the trained model."""
        print(f"Loading model from {self.params['model_path']}...")
        try:
            self.model = InteractionClassifier()
            self.model.load_state_dict(torch.load(self.params["model_path"], map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"Model loaded successfully and moved to {self.device}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def _load_embeddings(self):
        """Load protein embeddings."""
        print("Loading protein embeddings...")
        try:
            self.human_embeddings, self.yeast_embeddings = load_embeddings(
                self.params["human_embeddings_path"],
                self.params["yeast_embeddings_path"],
                node_column="node"
            )
            print(
                f"Loaded {len(self.human_embeddings)} human embeddings and {len(self.yeast_embeddings)} yeast embeddings")
        except Exception as e:
            raise RuntimeError(f"Error loading embeddings: {e}")

    def _load_orthologs(self):
        """Load ortholog mappings."""
        print("Loading ortholog mappings...")
        try:
            self.yeast_to_human, self.human_to_yeast = load_ortholog_mapping(self.params["ortholog_mapping_path"])
            print(f"Loaded {len(self.human_to_yeast)} human-to-yeast mappings")
        except Exception as e:
            raise RuntimeError(f"Error loading ortholog mappings: {e}")

    def _load_scores(self):
        """Load AlphaFold and average n models scores."""
        print("Loading AlphaFold and average n models scores...")
        try:
            self.af_scores = load_af_scores(self.params["af_scores_path"])
            self.avg_n_models = load_avg_n_models(self.params["avg_n_models_path"])
            print(f"Loaded {len(self.af_scores)} AlphaFold scores and {len(self.avg_n_models)} average n models scores")
        except Exception as e:
            raise RuntimeError(f"Error loading scores: {e}")

    def _predict_single_interaction(self, protein1: str, protein2: str) -> float:
        """Predict interaction probability for a single protein pair.

        Args:
            protein1: First protein identifier
            protein2: Second protein identifier

        Returns:
            float: Predicted interaction probability
        """
        # Get yeast orthologs
        yeast_protein1 = self.human_to_yeast.get(str(protein1))
        yeast_protein2 = self.human_to_yeast.get(str(protein2))

        # Get embeddings
        human_emb1 = self.human_embeddings.get(str(protein1), np.zeros(self.params["human_dim"]))
        human_emb2 = self.human_embeddings.get(str(protein2), np.zeros(self.params["human_dim"]))

        yeast_dim = self.params["yeast_dim"]
        yeast_emb1 = self.yeast_embeddings.get(yeast_protein1, np.zeros(yeast_dim)) if yeast_protein1 else np.zeros(
            yeast_dim)
        yeast_emb2 = self.yeast_embeddings.get(yeast_protein2, np.zeros(yeast_dim)) if yeast_protein2 else np.zeros(
            yeast_dim)

        # Create edge embeddings
        human_edge_emb = create_edge_embedding(human_emb1, human_emb2)
        yeast_edge_emb = create_edge_embedding(yeast_emb1, yeast_emb2)

        # Get scores
        af_score = self.af_scores.get((str(protein1), str(protein2)), 0.0)
        if not isinstance(af_score, (int, float)):
            try:
                af_score = float(af_score)
            except (ValueError, TypeError):
                print(f"Warning: Invalid AF score for {protein1}-{protein2}: {af_score}, defaulting to 0.0")
                af_score = 0.0

        avg_n_score = self.avg_n_models.get((str(protein1), str(protein2)), 0.0)
        if not isinstance(avg_n_score, (int, float)):
            try:
                avg_n_score = float(avg_n_score)
            except (ValueError, TypeError):
                print(
                    f"Warning: Invalid avg_n_models score for {protein1}-{protein2}: {avg_n_score}, defaulting to 0.0")
                avg_n_score = 0.0

        # Feature vector (4 features)
        feature_vector = np.array([
            human_edge_emb[0],  # Human similarity
            yeast_edge_emb[0],  # Yeast similarity
            af_score,  # AlphaFold score
            avg_n_score  # Average of n models
        ]).astype(np.float32)

        # Predict
        feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.model(feature_tensor)

        return prob.cpu().item()

    def predict_pair(self, protein1: str, protein2: str) -> Dict[str, Any]:
        """Predict interaction for a single protein pair.

        Args:
            protein1: First protein identifier
            protein2: Second protein identifier

        Returns:
            dict: Dictionary with prediction results
        """
        try:
            prob = self._predict_single_interaction(protein1, protein2)
            return {
                "protein1": protein1,
                "protein2": protein2,
                "Interaction_Probability": prob
            }
        except Exception as e:
            print(f"Error predicting for {protein1}-{protein2}: {e}")
            return {
                "protein1": protein1,
                "protein2": protein2,
                "Interaction_Probability": float('nan')
            }

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict interactions for protein pairs in a DataFrame.

        Args:
            df: DataFrame with 'protein1' and 'protein2' columns

        Returns:
            DataFrame: Original DataFrame with prediction results added
        """
        # Validate DataFrame columns
        if "protein1" not in df.columns or "protein2" not in df.columns:
            raise ValueError(f"DataFrame must contain 'protein1' and 'protein2' columns. Found: {df.columns.tolist()}")

        # Make predictions
        print(f"Predicting interactions for {len(df)} protein pairs...")
        results = []

        # Process in batches
        batch_size = min(self.params["batch_size"], 100)
        num_pairs = len(df)
        num_batches = (num_pairs + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_pairs)
            batch_df = df.iloc[start_idx:end_idx]

            for _, row in batch_df.iterrows():
                results.append(self.predict_pair(str(row["protein1"]), str(row["protein2"])))

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def predict_from_csv(self, input_csv: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Predict interactions for protein pairs in a CSV file.

        Args:
            input_csv: Path to CSV file with protein pairs (default: from config)

        Returns:
            DataFrame: DataFrame with prediction results
        """
        # Use input_csv from parameters if not provided
        if input_csv is None:
            input_csv = self.params["input_csv"]
        else:
            input_csv = Path(input_csv)

        # Load pairs from CSV
        print(f"Loading protein pairs from {input_csv}...")
        try:
            pairs_df = pd.read_csv(input_csv)
            print(f"Loaded {len(pairs_df)} protein pairs")
        except Exception as e:
            raise RuntimeError(f"Error loading CSV file: {e}")

        # Make predictions
        return self.predict_from_dataframe(pairs_df)

    def run(self, save_results: bool = True) -> pd.DataFrame:
        """Run predictions using parameters from config file.

        Args:
            save_results: Whether to save results to CSV file

        Returns:
            DataFrame: DataFrame with prediction results
        """

        # Make predictions
        results_df = self.predict_from_csv()

        # Save results if requested
        if save_results:
            output_path = self.output_dir / self.params["output_path"]
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")

        return results_df
