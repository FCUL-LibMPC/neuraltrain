import tyro
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import logging
import sys
import os

# Setup logger for CLI module
logger = logging.getLogger("neuraltrain.cli")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


@dataclass
class CLIArgs:
    """
    Command-line interface arguments for training a neural network model
    using Optuna to optimize hidden layer sizes.
    """

    input_file: str
    """Path to the training dataset CSV file."""

    train_id: str
    """Identifier for the Optuna study and W&B run."""

    output_dir: str
    """Directory to store training outputs and checkpoints."""

    processes: int = 1
    """Number of parallel processes to use for distributed training."""

    debug_mode: bool = False
    """Enable debug logging for more verbose output."""

    gpus: Optional[List[int]] = None
    """List of GPU device IDs to make visible. If not specified, all are used."""


def parse_cli_args() -> CLIArgs:
    args = tyro.cli(CLIArgs)

    if not args.debug_mode:
        logger.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)

    # Restrict visible GPUs if specified
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
        logger.info(
            f"Restricting visible GPUs to: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
    else:
        logger.info("No GPU restriction applied. Using all available GPUs.")

    # Validate process count
    if args.processes < 1:
        logger.error("Number of processes must be >= 1.")
        sys.exit(1)

    # Validate input dataset
    df_path = Path(args.input_file)
    if not df_path.exists():
        logger.error(f"Input file not found: {df_path}")
        sys.exit(1)
    if df_path.suffix.lower() != ".csv":
        logger.error(f"Invalid file type: {df_path}. Expected a CSV file.")
        sys.exit(1)

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return CLIArgs(
        input_file=Path(df_path),
        train_id=args.train_id,
        output_dir=Path(output_dir),
        processes=args.processes,
        debug_mode=args.debug_mode,
        gpus=args.gpus,
    )
