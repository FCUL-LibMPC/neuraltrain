# Include src folder to Python system path
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import json
import torch
import optuna
import pandas as pd
from pathlib import Path
from itertools import product
from neuraltrain import (
    NeuralNetworkTrainFramework, 
    FeedForwardNN, 
    TimeSeriesDataset,
    RobustnessEvalDataset,
    StudyConfig, 
    ModelConfig, 
    RunState
)
from typing import Dict, List, Tuple
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError

class TestTrainingFramework(NeuralNetworkTrainFramework):
    def __init__(self, study_config, model_config, dataset_df: pd.DataFrame, minmax_dict: dict, input_cols: List[str], target_cols: List[str], lag_config: Dict[str, List[int]]):
        super().__init__(study_config, model_config)
        self.df = dataset_df.copy()
        self.minmax_dict = minmax_dict
        self.input_cols = input_cols.copy()
        self.target_cols = target_cols.copy()
        self.lag_config = lag_config.copy()

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone()
        for i, col in enumerate(self.target_cols):
            feature_name = col.split("_")[0]
            min_val, max_val = self.minmax_dict[feature_name]["min"], self.minmax_dict[feature_name]["max"]
            tensor[:, i] = tensor[:, i] * (max_val - min_val) + min_val
        return tensor
    
    def apply_lag_mask(
        self,
        input_tensor: torch.Tensor,  # shape: [BATCH_ROWS, TIMESTEPS, VARIABLES]
        lag_config: Dict[str, List[int]],
        input_columns: List[str],
    ) -> torch.Tensor:
        """
        Builds a lag-masked input tensor from [BATCH_ROWS, TIMESTEPS, VARIABLES] using lag_config.

        Returns:
            Tensor of shape [BATCH_ROWS, LAGGED_INPUTS], where LAGGED_INPUTS is the total number of lags across all variables.
        """
        B, T, V = input_tensor.shape
        device = input_tensor.device
        masked_list = []

        for var_name in lag_config:
            if var_name not in input_columns:
                continue  # Skip unknown vars

            var_idx = input_columns.index(var_name)
            lags = lag_config[var_name]

            for lag in lags:
                time_idx = T - 1 - lag  # because T=historical_window, and we count from t=0 (last index)
                if time_idx < 0:
                    raise IndexError(f"Lag {lag} for variable '{var_name}' exceeds available window size {T}")

                masked_list.append(input_tensor[:, time_idx, var_idx].unsqueeze(1))

        return torch.cat(masked_list, dim=1)  # [BATCH_ROWS, LAGGED_INPUTS]

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
            # Mask lags to produce model-ready inputs; Shape: [BATCH_ROWS, LAGGED_INPUTS]
            masked_input = self.apply_lag_mask(inputs, self.lag_config, self.input_cols)

            outputs = model(masked_input)

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
            "Tz_mse_real": real_mse_metric.compute()[0].item(),
            "Tz_mae_real": real_mae_metric.compute()[0].item(),
            "Ehvac_mse_real": real_mse_metric.compute()[1].item(),
            "Ehvac_mae_real": real_mae_metric.compute()[1].item(),
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
            # Mask lags to produce model-ready inputs; Shape: [BATCH_ROWS, LAGGED_INPUTS]
            masked_input = self.apply_lag_mask(inputs, self.lag_config, self.input_cols)

            outputs = model(masked_input)

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

        for input_history, targets, input_size in self.robust_dl:
            input_size = input_size[0].item()  # Get the input size from the batch, equal to all rows
            predictions = torch.zeros(targets.shape, device=device)     # Shape: [BATCH_ROWS, PREDICTION_WINDOW, VARIABLES]
            tz_input_idx = self.input_cols.index("Tz")
            tz_output_idx = self.target_cols.index("Tz")
            for i in range(int(targets.shape[1])):
                # Mask lags to produce model-ready inputs; Shape: [BATCH_ROWS, LAGGED_INPUTS]
                masked_input = self.apply_lag_mask(input_history[:, i:i + input_size, :], self.lag_config, self.input_cols)
                outputs = model(masked_input)

                # Store predicted values to predictions tensor
                predictions[:, i, :] = outputs

                # Update Tz in the inputs t+1 for the next iteration
                input_history[:, i + 1, tz_input_idx] = outputs[:, tz_output_idx]

            
        # Calculate MAE for each timestep
        mae_per_timestep = torch.mean(torch.abs(predictions - targets), dim=0)  # [TIMESTEPS, MAE_VALUES]

        # Step 1: Store the MAE curve directly as a metric (e.g. for wandb or dict)
        mae_curve = mae_per_timestep.cpu().tolist()  # Convert to nested list: [[var1@t0, var2@t0], [var1@t1, var2@t1], ...]

        # Step 2: Average across variables to obtain a single curve
        average_mae_curve = torch.mean(mae_per_timestep, dim=1)  # Shape: [PREDICTION_TIMESTEPS]

        # Step 3: Compute integral (e.g. sum or trapezoidal rule for better approximation)
        integral_mae = torch.trapz(average_mae_curve).item()  # Scalar float

        # Final aggregation
        metrics = {
            "loss": total_loss / total_samples,
            "mse": mse_metric.compute().mean().item(),
            "mae": mae_metric.compute().mean().item(),
            "r2": r2_metric.compute().item(),
            "Tz_mse_real": real_mse_metric.compute()[0].item(),
            "Tz_mae_real": real_mae_metric.compute()[0].item(),
            "Ehvac_real": real_mse_metric.compute()[1].item(),
            "Ehvac_mae_real": real_mae_metric.compute()[1].item(),
            "robustness/mae": mae_curve,  # Full matrix of shape [timesteps, variables]
            "robustness/combined_mae": average_mae_curve.tolist(),  # List of floats
            "robustness/integral": integral_mae  # Final selection metric
        }

        return integral_mae, metrics
    
    def prepare_run(self, run: optuna.Trial, state: RunState, device: torch.device) -> None:
        def _get_first_tensor_device(dataloader):
            try:
                for batch in dataloader:
                    inputs, _ = batch
                    return inputs.device
            except StopIteration:
                return None
            except Exception:
                return None
        if hasattr(self, "train_dl") and hasattr(self, "test_dl") and hasattr(self, "robust_dl"):
            train_device = _get_first_tensor_device(self.train_dl)
            if train_device == device:
                self.logger.info("Reusing existing dataloaders; correct device detected.")
                return
        
        self.logger.info(f"Loading training and testing datasets with {state.train_proportion * 100}% training data...")
        self.train_dl, self.test_dl, _, _ = TimeSeriesDataset.get_dataloaders(
            self.df,
            period=300,
            input_cols=self.input_cols,
            target_cols=self.target_cols,
            input_window_size=24,
            train_proportion=state.train_proportion,
            device=device,
        )

        self.logger.info("Loading robust dataset...")
        self.robust_dl, self.robust_dataset = RobustnessEvalDataset.get_dataloader(
            df=df,
            period=300,
            input_cols=self.input_cols,
            target_cols=self.target_cols,
            step_size=1,                # 1 hour step size
            input_window_size=24,       # 1 day of historical data
            prediction_window_size=24,  # 1 day of prediction
            analysis_horizon=183,       # 6 months of data
        )
    
if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent

    # Load dataset and min-max normalization dictionary
    df = pd.read_csv(CURRENT_DIR / "library2024_simulation_dataset_normalized.csv")

    with open(CURRENT_DIR / "library2024_simulation_dataset_minmax.json", "r") as f:
        minmax_dict = pd.read_json(f)
        
    with open(CURRENT_DIR / "library2024_simulation_dataset_lags.json", "r") as f:
            lag_config = json.load(f)

    # Get the input and target columns (ORDERED)
    input_cols = ["Tz", "Tout", "WindSpd", "GHI", "Thvac", "RTUsouth", "win1", "win2", "win3", "win5"]
    target_cols = ["Tz", "Ehvac"]

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
        input_dim=sum(len(lags) for lags in lag_config.values()),
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
        lag_config=lag_config,
    )
    training_framework.run()
