import json
from pathlib import Path
from typing import Union, Dict, Any, Set
from argparse import Namespace


class ContactsAnalyzerDefaultParameters:
    """Defines the default parameters for protein contact analysis."""

    # Default config parameters used unless specified by user
    _defaults = {
        "mode": "batch",  # Mode: 'folder' or 'batch'
        "folder_path": None,  # Path to folder with PDB files (for 'folder' mode)
        "parent_dir": None,  # Path to parent directory containing folders with PDB files (for 'batch' mode)
        "output_format": "text",  # Output format for folder mode: 'text' or 'csv'
        "output_file": "contacts_summary.csv",  # Output file for batch mode
        "detailed": False,  # Whether to save detailed contact information
        "cutoff": 8.0,  # Distance cutoff in Angstroms to define a contact
        "pdb_prefix": "ranked_",  # Prefix for PDB files
        "pdb_indices": [0, 1, 2]  # Indices for PDB files to analyze
    }

    # Required parameters based on mode
    _mode_required_params = {
        "folder": {"folder_path"},
        "batch": {"parent_dir"}
    }

    # Parameters that should be cast to `Path` type
    path_args = {
        "folder_path",
        "parent_dir"
    }

    def __init__(self, args: Dict[str, Any]):
        """Initialize with processed arguments."""
        # Convert string paths to Path objects
        for arg in ContactsAnalyzerDefaultParameters.path_args:
            if arg not in args or args[arg] is None:
                continue

            args[arg] = Path(args[arg]).expanduser()

        self.args = args


class ContactsAnalyzerArgsParser(ContactsAnalyzerDefaultParameters):
    def __init__(self, args: Union[Path, Dict[str, Any], str]):
        """Parses and validates passed arguments.

        Args:
            args (Union[Path, Dict[str, Any], str]): Path to the parameter file or preloaded parameter
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
    def args(self, args: Union[Path, Dict[str, Any], str]) -> None:
        """Set and validate arguments.

        Args:
            args: Configuration file path or dictionary with parameters
        """
        # If `args` is a `Path` or string, load the JSON file
        if isinstance(args, (str, Path)):
            args_path = Path(args).expanduser()
            try:
                with open(args_path, 'r') as f:
                    args = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file not found: {args_path}")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in configuration file: {args_path}")

        # Filter out comment keys (keys starting with #)
        if isinstance(args, dict):
            args = {k: v for k, v in args.items() if not k.startswith('#')}

        # Get mode and validate required parameters for that mode
        mode = args.get("mode", ContactsAnalyzerDefaultParameters._defaults["mode"])
        self._validate_required_params(args, mode)

        self._args = args  # Store in private attribute

    def _validate_required_params(self, args: Dict[str, Any], mode: str) -> None:
        """Validate that all required parameters for the specified mode are present.

        Args:
            args: Dictionary of parameters to validate
            mode: The mode ('folder' or 'batch')

        Raises:
            ValueError: If any required parameters are missing or mode is invalid
        """
        if mode not in ContactsAnalyzerDefaultParameters._mode_required_params:
            raise ValueError(f"Invalid mode: '{mode}'. Supported modes are 'folder' and 'batch'.")

        required_params = ContactsAnalyzerDefaultParameters._mode_required_params[mode]
        missing_params = required_params - set(args.keys())

        if missing_params:
            missing_params_str = "`, `".join(missing_params)
            raise ValueError(
                f"Required parameter(s) `{missing_params_str}` not found in provided "
                f"parameter file for mode '{mode}'."
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
            Namespace: A Namespace object containing parsed parameters.
        """
        parsed_params = {
            param: self.get_parameters(param, default)
            for param, default in ContactsAnalyzerDefaultParameters._defaults.items()
        }

        namespace = Namespace(**parsed_params)
        return namespace