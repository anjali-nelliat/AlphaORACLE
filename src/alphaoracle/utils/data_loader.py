import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


# Load ortholog mapping
def load_ortholog_mapping(file_path='Ortholog_mapping.csv'):
    """Load the ortholog mapping between yeast and human proteins."""
    ortholog_df = pd.read_csv(file_path)
    # Assuming the first column is yeast protein and second is human ortholog
    # Convert all protein IDs to strings for consistent dictionary keys
    yeast_to_human = {str(row[0]): str(row[1]) for _, row in ortholog_df.iterrows()}
    human_to_yeast = {str(row[1]): str(row[0]) for _, row in ortholog_df.iterrows()}

    print(f"Loaded {len(yeast_to_human)} ortholog mappings")
    return yeast_to_human, human_to_yeast


def load_embeddings(human_embedding_file, yeast_embedding_file, node_column="node"):
    """
    Load node embeddings for human and yeast networks.

    Args:
        human_embedding_file: Path to human embeddings file
        yeast_embedding_file: Path to yeast embeddings file
        file_format: Format of the embedding files ("csv" or "txt")
        node_column: Column name that contains protein IDs (for CSV format)

    Returns:
        Tuple of dictionaries mapping protein IDs to embedding vectors
    """
    human_embeddings = {}
    yeast_embeddings = {}

    # Load from CSV
    try:
        human_df = pd.read_csv(human_embedding_file)
        yeast_df = pd.read_csv(yeast_embedding_file)

        # Check if node column exists
        if node_column not in human_df.columns:
            raise ValueError(
                f"Column '{node_column}' not found in human embeddings file. Available columns: {human_df.columns.tolist()}")
        if node_column not in yeast_df.columns:
            raise ValueError(
                f"Column '{node_column}' not found in yeast embeddings file. Available columns: {yeast_df.columns.tolist()}")

        # Get embedding columns (all columns except the node column)
        human_emb_cols = [col for col in human_df.columns if col != node_column]
        yeast_emb_cols = [col for col in yeast_df.columns if col != node_column]

        # Convert to dictionary mapping protein IDs to embedding vectors
        for _, row in human_df.iterrows():
            protein = str(row[node_column])
            human_embeddings[protein] = np.array(row[human_emb_cols].values, dtype=float)

        for _, row in yeast_df.iterrows():
            protein = str(row[node_column])
            yeast_embeddings[protein] = np.array(row[yeast_emb_cols].values, dtype=float)

    except Exception as e:
        print(f"Error loading CSV embeddings: {e}")
        raise

    print(
        f"Loaded {len(human_embeddings)} human protein embeddings and {len(yeast_embeddings)} yeast protein embeddings")

    # Get the dimensions from the first embedding in each dictionary
    human_dim = next(iter(human_embeddings.values())).shape[0] if human_embeddings else 0
    yeast_dim = next(iter(yeast_embeddings.values())).shape[0] if yeast_embeddings else 0

    print(f"Human embedding dimension: {human_dim}")
    print(f"Yeast embedding dimension: {yeast_dim}")

    return human_embeddings, yeast_embeddings, human_dim, yeast_dim


# Load AlphaFold confidence scores
def load_af_scores(file_path='AF_scores.csv'):
    """Load AlphaFold confidence scores for protein pairs."""
    af_scores_df = pd.read_csv(file_path)
    # Assuming the file has protein1, protein2, and confidence_score columns
    protein_pairs_to_af = {}
    for _, row in af_scores_df.iterrows():
        protein1, protein2 = str(row[0]), str(row[1])
        # Ensure the score is converted to float
        try:
            score = float(row[2])
        except (ValueError, TypeError):
            print(f"Warning: Invalid score for protein pair ({protein1}, {protein2}): {row[2]}, defaulting to 0.0")
            score = 0.0

        # Store both orders of the pair
        protein_pairs_to_af[(protein1, protein2)] = score
        protein_pairs_to_af[(protein2, protein1)] = score

    print(f"Loaded {len(protein_pairs_to_af) // 2} unique protein pairs with AlphaFold scores")
    return protein_pairs_to_af


def load_avg_n_models(file_path='avg_n_models.csv'):
    """Load average n models scores for protein pairs."""
    avg_n_models_df = pd.read_csv(file_path)
    # Assuming the file has protein1, protein2, and avg_n_models columns
    protein_pairs_to_avg_n_models = {}
    for _, row in avg_n_models_df.iterrows():
        protein1, protein2 = str(row[0]), str(row[1])
        # Ensure the score is converted to float
        try:
            score = float(row[2])
        except (ValueError, TypeError):
            print(
                f"Warning: Invalid avg_n_models score for protein pair ({protein1}, {protein2}): {row[2]}, defaulting to 0.0")
            score = 0.0

        # Store both orders of the pair
        protein_pairs_to_avg_n_models[(protein1, protein2)] = score
        protein_pairs_to_avg_n_models[(protein2, protein1)] = score

    print(f"Loaded {len(protein_pairs_to_avg_n_models) // 2} unique protein pairs with avg_n_models scores")
    return protein_pairs_to_avg_n_models


# Create edge embeddings from node embeddings using cosine similarity
def create_edge_embedding(node1_embedding, node2_embedding):
    """
    Create edge embedding from node embeddings using cosine similarity.
    Returns a scalar value representing similarity between the two nodes.
    """
    # Calculate cosine similarity (1 - cosine distance)
    # Handle zero vectors to avoid division by zero
    if np.all(node1_embedding == 0) or np.all(node2_embedding == 0):
        similarity = 0.0
    else:
        # cosine distance is 1 - cosine similarity, so we use 1 - cosine distance
        similarity = 1 - cosine(node1_embedding, node2_embedding)

    # Return as a single-element array to maintain compatibility with the rest of the code
    return np.array([similarity])
