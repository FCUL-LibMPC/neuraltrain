import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from neuraltrain import NeuralTrainerBase, parse_cli_args, TimeSeriesDataset
import torch

if __name__ == "__main__":
    args = parse_cli_args()


# from utils.wandb_api import generate_wandb_name
from neuraltrain.preprocessing import denormalize_temp, denormalize_mse
from typing import Dict, Any, List
import optuna
from neuraltrain.models import FeedForwardNN
import pandas as pd
from datetime import datetime


class OptunaHiddenLayers(NeuralTrainerBase):
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        return denormalize_temp(data)

    def input_col_filter(self, col: str) -> bool:
        return "Tout" in col and not col.endswith("_1")

    def target_col_filter(self, col: str) -> bool:
        return col == "Tout_1"

    def setup_trial(
        self,
        trial: optuna.Trial,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] | int,
        batch_sizes: List[int] | int,
        epochs: List[int] | int,
        train_proportion: List[float] | float,
        learning_rate: List[float] | float,
        weight_decay: List[float] | float,
    ) -> Dict[str, Any]:
        # Sample hyperparameters
        hidden_dim = hidden_dims[trial.number % len(hidden_dims)]
        bs = batch_sizes[trial.number // len(hidden_dims)]
        ep = (
            epochs
            if isinstance(epochs, int)
            else trial.suggest_categorical("epochs", epochs)
        )
        tp = (
            train_proportion
            if isinstance(train_proportion, float)
            else trial.suggest_float("train_prop", *train_proportion)
        )
        lr = (
            learning_rate
            if isinstance(learning_rate, float)
            else trial.suggest_float("lr", *learning_rate, log=True)
        )
        wd = (
            weight_decay
            if isinstance(weight_decay, float)
            else trial.suggest_float("weight_decay", *weight_decay)
        )

        # Generate checkpoint path and wandb info
        checkpoint_path = (
            self.output_dir
            / "checkpoints"
            / f"trial_{trial.number}_hl_{hidden_dim}_bs_{bs}.pkl"
        )
        wandb_id = f"hd_{hidden_dim}_bs_{bs}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        wandb_name = f"Hidden_{hidden_dim}_BS_{bs}_T{trial.number}"

        return {
            "hidden_dims": [hidden_dim, hidden_dim],
            "batch_size": bs,
            "epochs": ep,
            "train_proportion": tp,
            "learning_rate": lr,
            "weight_decay": wd,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "checkpoint_path": checkpoint_path,
            "wandb_id": wandb_id,
            "wandb_name": wandb_name,
        }


if __name__ == "__main__":
    # Neural Network parameters
    hidden_dims = [10, 50]

    # Dataset parameters
    batch_sizes = [64, 128, 512, 1024, 2048, 4096, 6144, 8192]
    train_proportion = 0.8

    # Optimizer parameters
    epochs = 2000
    learning_rate = 0.001
    weight_decay = learning_rate / epochs

    # Load CSV dataset and Pytorch dataloaders
    df = pd.read_csv(args.input_file)
    model_var = "Tout"
    train_dl, test_dl = TimeSeriesDataset.get_dataloaders(
        df,
        input_col_filter=lambda col: model_var in col and col != f"{model_var}_1",
        target_col_filter=lambda col: col == f"{model_var}_1",
        train_proportion=train_proportion,
        batch_size=batch_sizes[0],
    )

    # Get input and target layers dimensions
    input_dim = len(train_dl.dataset.input_cols)
    output_dim = len(train_dl.dataset.target_cols)

    # Avoid reusing dataset in the jobs processes
    del train_dl, test_dl, df

    # Start the train
    study = OptunaHiddenLayers(
        db_url="sqlite:///optuna_study.db",
        study_name=args.train_id,
        n_trials=len(hidden_dims) * len(batch_sizes),
        output_dir=args.output_dir,
        model=FeedForwardNN,
        optimizer=torch.optim.Adam,
        debug_mode=args.debug_mode,
    )
    study.distributed_run(
        args.processes,
        hidden_dims,
        input_dim,
        output_dim,
        epochs,
        learning_rate,
        weight_decay,
        args.input_file,
        train_proportion,
        batch_sizes,
        args.output_dir,
        args.train_id,
    )

    # Import data from WandB to a CSV
    import_from_wandb(study_name, output_dir)
