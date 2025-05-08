import json
from pathlib import Path
from typing import Union, Dict, Any, Set
from argparse import Namespace


class ClassifierDefaultParameters:
    """Defines the default parameters for AlphaOracle protein interaction prediction."""

    # Default config parameters used unless specified by user
    _defaults = {
        "ortholog_mapping_path": "AlphaORACLE/inputs/Ortholog_mapping.csv",  # Path to ortholog mapping file
        "human_embeddings_path": "AlphaORACLE/inputs/embeddings/human_features.csv",  # Path to human embeddings file
        "yeast_embeddings_path": "AlphaORACLE/inputs/embeddings/yeast_features.csv",  # Path to yeast embeddings file
        "af_scores_path": None,  # Path to AlphaFold scores file
        "avg_n_models_path": None,  # Path to average models file
        "training_data_path": None,  # Path to training data file
        "n_trials": 30,  # Number of Optuna optimization trials
        "timeout": 7200,  # Timeout in seconds for optimization
        "epochs": 30,  # Maximum number of training epochs
        "patience": 10,  # Early stopping patience
        "output_dir": "./alphaoracle_output",  # Directory to save outputs
        "model_file": "AlphaORACLE/inputs/optimized_interaction_classifier.pt",  # Filename for saved model
        "study_file": "optuna_study.pkl",  # Filename for optimization study
        "predictions_file": "optimized_test_predictions.csv",  # Filename for test predictions
        "performance_plot": "model_performance.png",  # Filename for performance plot
        "human_dim": 1024,  # Dimension of human protein embeddings
        "yeast_dim": 512,  # Dimension of yeast protein embeddings
        "train_test_split_ratio": 0.25,  # Ratio for test:train split
        "train_val_split_ratio": 0.25,  # Ratio for validation:train split
        "random_seed": 42,  # Random seed for reproducibility
        "optimization_history_plot": "optimization_history.png",  # Filename for optimization history plot
        "param_importances_plot": "param_importances.png",  # Filename for parameter importances plot
        "training_history_plot": "training_history.png",  # Filename for training history plot
    }

    # Required parameters not specified in `_defaults`
    required_params = {
        "ortholog_mapping_path",
        "human_embeddings_path",
        "yeast_embeddings_path",
        "af_scores_path",
        "avg_n_models_path",
        "training_data_path"
    }

    # Parameters that should be cast to `Path` type
    path_args = {
        "ortholog_mapping_path",
        "human_embeddings_path",
        "yeast_embeddings_path",
        "af_scores_path",
        "avg_n_models_path",
        "training_data_path",
        "output_dir"
    }

    def __init__(self, args: Dict[str, Any]):
        """Initialize with processed arguments."""
        # Convert string paths to Path objects
        for arg in ClassifierDefaultParameters.path_args:
            if arg not in args or args[arg] is None:
                continue

            args[arg] = Path(args[arg]).expanduser()

        self.args = args


class ClassifierArgsParser(ClassifierDefaultParameters):
    def __init__(self, args: Union[Path, Dict[str, Any]]):
        """Parses and validates passed arguments.

        Args:
            args (Union[Path, Dict[str, Any]]): Path to the parameter file or preloaded parameter
                dictionary.
        """
        self._args = None  # Initialize private attribute
        self.args = args  # Use the setter to process args
        super().__init__(self._args)  # Pass processed args to parent

    @property
    def args(self) -> Dict[str, Any]:
        """Get the processed arguments."""
        return self._args

    @args.setter
    def args(self, args: Union[Path, Dict[str, Any]]) -> None:
        """Set and validate arguments.

        Args:
            args: Configuration file path or dictionary with parameters
        """
        # If `args` is a `Path`, load the JSON file
        if isinstance(args, (str, Path)):
            args_path = Path(args).expanduser()
            try:
                with open(args_path, 'r') as f:
                    args = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file not found: {args_path}")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in configuration file: {args_path}")

            # Set output_dir to config file's parent directory if not specified
            if "output_dir" not in args:
                args["output_dir"] = args_path.parent / "alphaoracle_output"

        # Filter out comment keys (keys starting with #)
        if isinstance(args, dict):
            args = {k: v for k, v in args.items() if not k.startswith('#')}

        # Validate required parameters
        self._validate_required_params(args)

        self._args = args  # Store in private attribute

    def _validate_required_params(self, args: Dict[str, Any]) -> None:
        """Validate that all required parameters are present.

        Args:
            args: Dictionary of parameters to validate

        Raises:
            ValueError: If any required parameters are missing
        """
        missing_params = ClassifierDefaultParameters.required_params - set(args.keys())
        if missing_params:
            missing_params_str = "`, `".join(missing_params)
            raise ValueError(
                f"Required parameter(s) `{missing_params_str}` not found in provided "
                "parameter file."
            )

    def get_parameters(self, param: str, default: Any) -> Any:
        """Get parameter value from args or return default.

        Args:
            param: Parameter name
            default: Default value to return if parameter not in args

        Returns:
            Parameter value from args or default
        """
        return self._args.get(param, default)

    def parse(self) -> Namespace:
        """Parse parameter file.

        Overrides default params with user provided params and returns them
        as a namespace.

        Returns:
            Namespace: A Namespace object containing parsed AlphaOracle parameters.
        """
        parsed_params = {
            param: self.get_parameters(param, default)
            for param, default in ClassifierDefaultParameters._defaults.items()
        }

        namespace = Namespace(**parsed_params)
        return namespace


class PredictionDefaultParameters:
    """Defines the default parameters for protein interaction prediction."""

    # Default config parameters used unless specified by user
    _defaults = {
        "ortholog_mapping_path": "AlphaORACLE/inputs/Ortholog_mapping.csv",  # Path to ortholog mapping file
        "human_embeddings_path": "AlphaORACLE/inputs/embeddings/human_features.csv",  # Path to human embeddings file
        "yeast_embeddings_path": "AlphaORACLE/inputs/embeddings/yeast_features.csv",  # Path to yeast embeddings file
        "af_scores_path": None,  # Path to AlphaFold scores file
        "avg_n_models_path": None,  # Path to average models file
        "input_csv": None,  # Path to input CSV with protein pairs to predict
        "model_path": "AlphaORACLE/inputs/optimized_interaction_classifier.pt",  # Path to trained model
        "output_path": "interaction_predictions.csv",  # Path to save predictions
        "output_dir": "./prediction_output",  # Directory to save outputs
        "yeast_dim": 512,  # Dimension of yeast protein embeddings
        "human_dim": 1024,  # Dimension of human protein embeddings
        "batch_size": 32,  # Batch size for predictions
    }

    # Required parameters not specified in `_defaults`
    required_params = {
        "ortholog_mapping_path",
        "human_embeddings_path",
        "yeast_embeddings_path",
        "af_scores_path",
        "avg_n_models_path",
        "input_csv"
    }

    # Parameters that should be cast to `Path` type
    path_args = {
        "ortholog_mapping_path",
        "human_embeddings_path",
        "yeast_embeddings_path",
        "af_scores_path",
        "avg_n_models_path",
        "input_csv",
        "model_path",
        "output_path",
        "output_dir"
    }

    def __init__(self, args: dict):
        """Initialize with processed arguments."""
        # Convert string paths to Path objects
        for arg in PredictionDefaultParameters.path_args:
            if arg not in args or args[arg] is None:
                continue

            args[arg] = Path(args[arg]).expanduser()

        self.args = args


class PredictionArgsParser(PredictionDefaultParameters):
    def __init__(self, args_path):
        """Parses and validates passed arguments.

        Args:
            args_path: Path to the parameter file.
        """
        self._args = None  # Initialize private attribute
        self.args = args_path  # Use the setter to process args
        super().__init__(self._args)  # Pass processed args to parent

    @property
    def args(self):
        """Get the processed arguments."""
        return self._args

    @args.setter
    def args(self, args_path):
        """Set and validate arguments.

        Args:
            args_path: Configuration file path
        """
        # Load the JSON file
        args_path = Path(args_path).expanduser()
        try:
            with open(args_path, 'r') as f:
                args = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {args_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in configuration file: {args_path}")

        # Set output_dir to config file's parent directory if not specified
        if "output_dir" not in args:
            args["output_dir"] = args_path.parent / "prediction_output"

        # Filter out comment keys (keys starting with #)
        args = {k: v for k, v in args.items() if not k.startswith('#')}

        # Validate required parameters
        self._validate_required_params(args)

        self._args = args  # Store in private attribute

    def _validate_required_params(self, args):
        """Validate that all required parameters are present."""
        missing_params = PredictionDefaultParameters.required_params - set(args.keys())
        if missing_params:
            missing_params_str = "`, `".join(missing_params)
            raise ValueError(
                f"Required parameter(s) `{missing_params_str}` not found in provided "
                "parameter file."
            )

    def get_parameters(self, param, default):
        """Get parameter value from args or return default."""
        return self._args.get(param, default)

    def parse(self):
        """Parse parameter file.

        Overrides default params with user provided params and returns them
        as a dictionary.

        Returns:
            dict: Dictionary containing parsed parameters.
        """
        parsed_params = {
            param: self.get_parameters(param, default)
            for param, default in PredictionDefaultParameters._defaults.items()
        }

        namespace = Namespace(**parsed_params)
        return namespace
