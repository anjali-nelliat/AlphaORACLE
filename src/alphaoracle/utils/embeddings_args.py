import json
from pathlib import Path
from typing import Union, List, Any
from argparse import Namespace


class EmbeddingsDefaultParameters:
    """Defines the default parameters for integrating networks using MultiviewGAT."""

    # Default config parameters used unless specified by user
    _defaults = {
        "input_names": None,  # Input file path
        "output_name": None,  # Name of output files
        "delimiter": " ",  # Delimiter for network input files
        "epochs": 1000,  # Number of training epochs
        "batch_size": 2048,  # Number of genes/proteins in each batch
        "learning_rate": 0.0005,  # Adam optimizer learning rate
        "embedding_size": 512,  # Dimensionality of output integrated features
        "neighbor_sample_size": 2,  # Number of neighbors to sample per node each epoch
        "gat_shapes": {
            "dimension": 128,  # Dimension of each GAT layer
            "n_heads": 10,  # Number of attention heads for each GAT layer
            "n_layers": 3,  # Number of GAT layers for each input network
        },
        "save_model": True,  # Whether to save the trained model or not
        "pretrained_model_path": None,  # Path to pretrained model state dict
        "plot_loss": True,  # Whether to plot loss curves
        "save_loss_data": True,  # Whether to save the training loss data in a .tsv file
    }

    # Required parameters not specified in `_defaults`
    required_params = {"input_names"}

    # Parameters that should be cast to `Path` type
    path_args = {"input_names", "output_name"}

    def __init__(self, args: dict):
        # Make sure all path strings are mapped to `Path`s
        for arg in EmbeddingsDefaultParameters.path_args:
            if arg not in args:
                continue

            if isinstance(args[arg], list):
                args[arg] = [
                    Path(path_string).expanduser() for path_string in args[arg]
                ]
            elif args[arg] is None:
                args[arg] = None
            else:
                args[arg] = Path(args[arg]).expanduser()

        self.args = args


class EmbeddingsArgsParser(EmbeddingsDefaultParameters):
    def __init__(self, args: Union[Path, dict]):
        """Parses and validates passed arguments

        Args:
            args (Union[Path, dict]): Name of the parameter file or preloaded parameter
                dictionary.
        """
        self._args = None  # Initialize private attribute
        self.args = args  # Use the setter to process args
        super().__init__(self._args)  # Pass processed args to parent

    @property
    def args(self):
        return self._args  # Return the private attribute

    @args.setter
    def args(self, args: Union[Path, dict]) -> None:
        # Check if `args` is already loaded and validate `output_name` exists if so
        if isinstance(args, dict) and "output_name" not in args:
            raise ValueError(
                "Output file name `output_name` must be provided in `args` if "
                "`args` is provided as a dictionary"
            )

        # If `args` is a `Path`, load and set `output_name` parameter if not specified
        if isinstance(args, Path):
            args_path = args
            with args_path.open() as f:
                args = json.load(f)  # args is now a dictionary if it was previously a `Path`

            if "output_name" not in args:
                args["output_name"] = args_path.parent / args_path.stem

        # Validate required parameters are present in `args`
        required_params = EmbeddingsDefaultParameters.required_params
        if len(required_params.intersection(set(args.keys()))) != len(required_params):
            missing_params = "`, `".join(EmbeddingsDefaultParameters.required_params - set(args.keys()))
            raise ValueError(
                f"Required parameter(s) `{missing_params}` not found in provided " "parameter file."
            )

        self._args = args  # Store in private attribute

    def resolve_path(self, path: Path) -> List[Path]:
        directory = path.parent
        return [p for p in directory.iterdir() if not p.is_dir()]

    def get_parameters(self, param: str, default: Any) -> Any:
        if param in self._args:  # Access private attribute
            value = self._args[param]

            # Handle `Path` versions of "names" parameter
            if param == "input_names" and isinstance(value, Path):
                if value.stem == "*":
                    return self.resolve_path(value)
                else:
                    return [value]  # Wrap path in a list to ensure compatibility

            return value
        else:
            return default

    def parse(self) -> Namespace:
        """Parses parameter file.

        Overrides default params with user provided params and returns them
        namespaced.

        Returns:
            Namespace: A Namespace object containing parsed MultiviewGAT parameters.
        """
        parsed_params = {
            param: self.get_parameters(param, default)
            for param, default in EmbeddingsDefaultParameters._defaults.items()
        }

        namespace = Namespace(**parsed_params)
        return namespace