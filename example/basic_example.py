# Include src folder to Python system path
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import optuna
import pandas as pd
from pathlib import Path
from itertools import product
from neuraltrain import (
    NeuralNetworkTrainFramework, 
    FeedForwardNN, 
    BasicDataset, 
    StudyConfig, 
    ModelConfig, 
    RunState
)
from typing import Dict, List, Tuple
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError

class TestTrainingFramework(NeuralNetworkTrainFramework):
    def __init__(self, study_config, model_config, dataset_df: pd.DataFrame, minmax_dict: dict, input_cols: List[str] = None, target_cols: List[str] = None):
        super().__init__(study_config, model_config)
        self.df = dataset_df.copy()
        self.minmax_dict = minmax_dict
        self.input_cols = input_cols.copy()
        self.target_cols = target_cols.copy()

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone()
        for i, col in enumerate(self.target_cols):
            feature_name = col.split("_")[0]
            min_val, max_val = self.minmax_dict[feature_name]["min"], self.minmax_dict[feature_name]["max"]
            tensor[:, i] = tensor[:, i] * (max_val - min_val) + min_val
        return tensor

    def train(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
        # Normalized metrics
        r2_metric = R2Score().to(device)
        mae_metric = MeanAbsoluteError(num_outputs=len(self.target_cols)).to(device)
        mse_metric = MeanSquaredError(num_outputs=len(self.target_cols)).to(device)

        # Denormalized metrics per output variable
        real_mae_metric = MeanAbsoluteError(num_outputs=len(self.target_cols)).to(device)
        real_mse_metric = MeanSquaredError(num_outputs=len(self.target_cols)).to(device)

        # Initialize loss variables
        total_loss = 0.0
        total_samples = 0

        model.train()

        for inputs, targets in self.train_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Backpropagation and optimization
            loss = torch.nn.MSELoss()(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate weighted loss
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Normalized updates
            r2_metric.update(outputs, targets)
            mae_metric.update(outputs, targets)
            mse_metric.update(outputs, targets)

            # Desnormalize (assumindo radiação entre 0 e 1000)
            outputs_real = self.denormalize(outputs)
            targets_real = self.denormalize(targets)

            # Denormalized updates
            real_mae_metric.update(outputs_real, targets_real)
            real_mse_metric.update(outputs_real, targets_real)

        # Final aggregation
        metrics = {
            "loss": total_loss / total_samples,
            "mse": mse_metric.compute().mean().item(),
            "mae": mae_metric.compute().mean().item(),
            "r2": r2_metric.compute().item(),
            "Tz_mse": real_mse_metric.compute()[0].item(),
            "Tz_mae": real_mae_metric.compute()[0].item(),
            "Ehvac": real_mse_metric.compute()[1].item(),
            "Ehvac_mae": real_mae_metric.compute()[1].item(),
        }

        return metrics
    
    def evaluate(self, model: torch.nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
        # Normalized metrics
        r2_metric = R2Score().to(device)
        mae_metric = MeanAbsoluteError(num_outputs=len(self.target_cols)).to(device)
        mse_metric = MeanSquaredError(num_outputs=len(self.target_cols)).to(device)

        # Denormalized metrics per output variable
        real_mae_metric = MeanAbsoluteError(num_outputs=len(self.target_cols)).to(device)
        real_mse_metric = MeanSquaredError(num_outputs=len(self.target_cols)).to(device)

        # Initialize loss variables
        total_loss = 0.0
        total_samples = 0

        model.eval()

        for inputs, targets in self.test_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = torch.nn.MSELoss()(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Normalized updates
            r2_metric.update(outputs, targets)
            mae_metric.update(outputs, targets)
            mse_metric.update(outputs, targets)

            # Denormalized updates
            outputs_real = self.denormalize(outputs)
            targets_real = self.denormalize(targets)
            real_mae_metric.update(outputs_real, targets_real)
            real_mse_metric.update(outputs_real, targets_real)

        # Final aggregation
        metrics = {
            "loss": total_loss / total_samples,
            "mse": mse_metric.compute().mean().item(),
            "mae": mae_metric.compute().mean().item(),
            "r2": r2_metric.compute().item(),
            "Tz_mse": real_mse_metric.compute()[0].item(),
            "Tz_mae": real_mae_metric.compute()[0].item(),
            "Ehvac": real_mse_metric.compute()[1].item(),
            "Ehvac_mae": real_mae_metric.compute()[1].item(),
        }

        return metrics["loss"], metrics
    
    def prepare_run(self, run: optuna.Trial, state: RunState, device: torch.device) -> None:
        self.train_dl, self.test_dl = BasicDataset.get_dataloaders(
            self.df,
            input_cols=self.input_cols,
            target_cols=self.target_cols,
            train_proportion=state.train_proportion
        )
    
if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent

    # Load dataset and min-max normalization dictionary
    df = pd.read_csv(CURRENT_DIR / "library2024_simulation_dataset_lagged.csv")
    with open(CURRENT_DIR / "library2024_simulation_dataset_minmax.json", "r") as f:
        minmax_dict = pd.read_json(f)

    # Get the input and target columns
    input_cols = [col for col in df.columns if not col.endswith("_target") and col != "Timestamp"]
    target_cols = [col for col in df.columns if col.endswith("_target")]

    # Define hyperparameter search space
    hidden_layer_sizes = [32, 64]
    batch_sizes = [16]
    epochs_list = [100]
    learning_rates = [1e-3]
    weight_decays = [1e-4]
    train_proportions = [0.8]

    # Build search space as a list of tuples with all combinations
    search_space = list(product(
        [(h1, h2) for h1 in hidden_layer_sizes for h2 in hidden_layer_sizes],
        batch_sizes,
        epochs_list,
        learning_rates,
        weight_decays,
        train_proportions
    ))

    # Configure the study
    study_config = StudyConfig(
        sqlite_url="sqlite:///test_study.db",
        study_id="test_run",
        search_space=search_space,
        output_dir= CURRENT_DIR / "test_output",
    )

    # Configure the model
    model_config = ModelConfig(
        model_class=FeedForwardNN,
        optimizer_class=torch.optim.Adam,
        input_dim=len(input_cols),
        output_dim=len(target_cols),
    )

    # Initialize the training framework
    training_framework = TestTrainingFramework(
        study_config=study_config,
        model_config=model_config,
        dataset_df=df,
        minmax_dict=minmax_dict,
        input_cols=input_cols,
        target_cols=target_cols,
    )
    training_framework.run()
