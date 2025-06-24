import optuna
import sys
import multiprocessing
from abc import ABC, abstractmethod
from typing import List
from optuna.study import MaxTrialsCallback
from pathlib import Path
import pickle
import sqlite3
import tempfile
from typing import List, Dict, Any
import logging
import torch
from .dataloaders import TimeSeriesDataset, DataLoader, Dataset
from torchmetrics.regression import MeanAbsoluteError, R2Score, MeanSquaredError
import pandas as pd
import wandb
from typing import Type, List
from enum import Enum
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from dataclasses import dataclass, replace

@dataclass
class RunConfig:
    hidden_dims: List[int]
    batch_size: int
    epochs: int
    train_proportion: float
    learning_rate: float
    weight_decay: float
    input_dim: int
    output_dim: int

@dataclass
class RunState:
    # Paths and IDs
    checkpoint_path: Path

    # Model configuration
    input_dim: int
    output_dim: int
    hidden_dims: List[int]

    # Optimizer configuration
    learning_rate: float
    weight_decay: float

    # Training configuration
    train_proportion: float
    batch_size: int
    epochs: int

    # Checkpointed runtime state
    epoch: int
    best_model_loss: float
    best_model_state_dict: Dict[str, Any]
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]

    # WandB configuration
    wandb_id: str
    wandb_name: str

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """
        Extracts the checkpoint-relevant training state from the RunState to store as pickled dictionary.
        Remaining information is not relevant for checkpointing as it is stored in the database.

        Returns:
            dict: A dictionary containing runtime training state for checkpointing.
        """
        return {
            "epoch": self.epoch,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "best_model_loss": self.best_model_loss,
            "best_model_state_dict": self.best_model_state_dict,
        }

class NoCheckpointAvailable(Exception):
    """Raised when no checkpoint is configured or found for a trial."""
    pass

class NeuralTrainerBase(ABC):
    """Abstract class to implement a Neural Network training levaraging Optuna for optimization and paralelism."""

    def __init__(
        self,
        db_url: str,
        study_id: str,
        n_runs: int,
        output_dir: str,
        model: Type[Module],
        optimizer: Type[Optimizer],
        output_mapping: List[str],
        sampler=None,
        pruner=None,
        debug_mode=False,
    ):
        # Set relevant global attributes to the entire study
        self.__db_url = db_url
        self.__study_id = study_id
        self.__n_runs = n_runs
        self.__output_dir = Path(output_dir) / f"{self.__study_id}"
        self.__debug_mode = debug_mode
        self.__model_class = model
        self.__optimizer_class = optimizer
        self.__output_mapping = output_mapping

        # Create output directory if it doesn't exist
        self.__output_dir.mkdir(parents=True, exist_ok=True)

        # Modified by distributed_run in case of multiple processes
        self.__n_processes = 1                  

        # Setup the logger
        self.__init_logger(self.__debug_mode)

        # Check and recover from previously failed/incomplete study
        all_runs = self.__get_all_runs()
        if all_runs:
            # Get completed runs
            completed_runs = [
                run for run in all_runs if run["state"] == "COMPLETE"
            ]
            self.logger.debug(
                f"Found {len(completed_runs)} COMPLETED run(s) in study '{self.__study_id}'."
            )

            # Get failed runs
            failed_runs = [
                run for run in all_runs if run["state"] in ["FAIL", "RUNNING"]
            ]
            self.logger.debug(
                f"Found {len(failed_runs)} FAIL/RUNNING run(s) in study '{self.__study_id}'."
            )

            # Recover study from failed state; Subtract completed runs from n_runs
            self.__n_runs -= len(completed_runs)

            # Re-enqueue failed runs
            if failed_runs:
                self.logger.warning(
                    f"Re-enqueuing {len(failed_runs)} failed run(s) in study '{self.__study_id}'."
                )
                self.__reenqueue_failed_trials(
                    [run["trial_id"] for run in failed_runs]
                )

        # Create a new study or load an existing one
        self.study = optuna.create_study(
            storage=self.__db_url,
            study_name=self.__study_id,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        self.logger.info(f"Study '{self.__study_id}' initialized successfully.")

    def __init_logger(self, debug_mode: bool = False):
        # Setting the log level based on debug mode
        if debug_mode:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO

        # Setting up the logger
        logger = logging.getLogger("NeuralTrainBase")
        logger.setLevel(logging_level)

        # Remove any existing handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()

        # Setting up the formatter for structured logging
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.propagate = False

        self.logger = logger

        self.logger.debug(f"Logger initialized for study '{self.__study_id}'.")

    def __init_run_logger(self, run: optuna.Trial):
        """
        Initializes a per-run logger with the run number included in the log format.

        Args:
            run_number (int): The Optuna trial number for this run.
        """
        logger_name = f"NeuralTrainBase.Run{run.number}"
        logger = logging.getLogger(logger_name)

        level = logging.DEBUG if self.__debug_mode else logging.INFO
        logger.setLevel(level)

        # Remove any existing handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter(
            f"[%(asctime)s] [RUN {run.number}] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        self.logger = logger

    def __reenqueue_failed_trials(self, trial_ids: List[int]) -> None:
        """Manually re-enqueues failed trials back to WAITING state in the Optuna SQLite database.

        Args:
            trial_ids (List[int]): A list of trial IDs to re-enqueue.
        """

        if not trial_ids:
            return

        # Extract the database path from the URL
        db_path = self.__db_url.replace("sqlite:///", "")

        # Check if the database file exists
        if not Path(db_path).exists():
            self.logger.warning(
                f"Database file not found at '{db_path}'. Expected if it is the first time."
            )
            return

        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Update the state of FAILED and RUNNING trials to WAITING
        cursor.executemany(
            """
            UPDATE trials
            SET state = ?, datetime_start = NULL, datetime_complete = NULL
            WHERE trial_id = ? AND state IN (?, ?)
            """,
            [("WAITING", trial_id, "FAIL", "RUNNING") for trial_id in trial_ids],
        )

        # Get the number of affected rows
        affected_rows = cursor.rowcount

        # Apply changes to the database and close SQLite database connection
        conn.commit()
        conn.close()

        self.logger.warning(
            f"Re-enqueued {affected_rows} failed trial(s) in study '{self.__study_id}'."
        )

    def __get_all_runs(self) -> List[Dict[str, Any]]:
        """Retrieves all trials from the Optuna SQLite database for the current study."""

        # Extract SQLite database path from the url
        db_path = self.__db_url.removeprefix("sqlite:///")

        # Check if database already exists, if not return an empty list
        if not Path(db_path).exists():
            self.logger.warning(
                f"No database found at '{db_path}'. Expected if it is the first time."
            )
            return []
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Retrieve the study_id
        result = cursor.execute(
            "SELECT study_id FROM studies WHERE study_name = ?", (self.study_id,)
        ).fetchone()

        # Check if the study exists
        if result is None:
            conn.close()
            self.logger.warning(
                f"Study '{self.__study_id}' not found in database. Expected if it is the first time."
            )
            return []
        study_id = result[0]

        # Fetch all trials for the study
        cursor.execute(
            """
            SELECT trial_id, number, state, datetime_start, datetime_complete
            FROM trials
            WHERE study_id = ?
            ORDER BY trial_id ASC
            """,
            (study_id,),
        )
        trials = cursor.fetchall()

        # Close SQLite database connection
        conn.close()

        # Convert trial data into a list of dictionaries
        trials_list = [dict(trial) for trial in trials]
        self.logger.debug(f"Retrieved {len(trials)} trials from the database.")
        if self.__debug_mode and trials_list:
            for trial in trials_list:
                self.logger.debug(trial)
        return trials_list

    def __load_checkpoint(self, checkpoint_path: Path) -> dict:
        """Loads a checkpoint from a specified path if it exists.

        Args:
            checkpoint_path (Path): The full path to the .pkl checkpoint file.

        Returns:
            dict: The checkpoint state if the file exists and is valid.
        """

        # Checking for existing checkpoints
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found at '{checkpoint_path}'.")
            return {}

        try:
            with checkpoint_path.open("rb") as f:
                state = pickle.load(f)
            self.logger.info(f"Loaded checkpoint from '{checkpoint_path}'.")
            return state
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from '{checkpoint_path}'.")
            self.logger.error(f"Pickle Error: {e}")
            return {}

    def __try_resume_from_checkpoint(self, run: optuna.Trial, device: torch.device) -> RunState:
        """
        Attempt to resume training from a previously saved checkpoint based on the run's user attributes.

        Raises:
            NoCheckpointAvailable: If no checkpoint information is found for the given trial.
            Exception: Possibly related to incomplete/corrupted checkpoints (e.g. KeyError for missing entries in attrs or checkpoint).
        """
        
        # Check if the run has an existing checkpoint
        if not run.user_attrs or "checkpoint_path" not in run.user_attrs or not Path(run.user_attrs["checkpoint_path"]).exists():
            raise NoCheckpointAvailable(f"No checkpoint found.")
        
        attrs = run.user_attrs

        self.logger.info(f"Checkpoint found. Attempting to load...")
        self.logger.debug(f"Checkpoint file found at: {run.user_attrs['checkpoint_path']}")

        # Get checkpoint file path
        checkpoint_path = Path(attrs["checkpoint_path"])

        # Get model parameters
        hidden_layers_sizes = attrs["hidden_dims"]
        input_dim = attrs["input_dim"]
        output_dim = attrs["output_dim"]

        # Get optimizer parameters
        learning_rate = attrs["learning_rate"]
        weight_decay = attrs["weight_decay"]
        
        # Get training parameters
        train_proportion = attrs["train_proportion"]
        batch_size = attrs["batch_size"]
        epochs = attrs["epochs"]

        # Get wandb parameters
        wandb_id = attrs["wandb_id"]
        wandb_name = attrs["wandb_name"]

        # Load checkpoint information
        checkpoint = self.__load_checkpoint(checkpoint_path)
        model_state_dict = checkpoint["model_state_dict"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        current_epoch = checkpoint["epoch"]
        best_model_loss = checkpoint["best_model_loss"]
        best_model_state_dict = checkpoint["best_model_state_dict"]

        self.logger.info(f"Checkpoint loaded.")

        return RunState(
            checkpoint_path=checkpoint_path,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_layers_sizes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_proportion=train_proportion,
            batch_size=batch_size,
            epochs=epochs,
            epoch=current_epoch,
            best_model_loss=best_model_loss,
            best_model_state_dict=best_model_state_dict,
            model_state_dict=model_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            wandb_id=wandb_id,
            wandb_name=wandb_name,
        )

    def __check_run_config(self, config: RunConfig) -> None:
        """Raise descriptive error for invalid values ir configuration."""

        # Check train proportion
        tp = config.train_proportion
        if not 0.0 < tp < 1.0:
            raise ValueError(
                f"Invalid train_proportion: {tp}. Must be between 0 and 1 (exclusive)."
            )
        
        # Check batchsize value
        bs = config.batch_size
        if not isinstance(bs, int) or bs <= 0:
            raise ValueError(f"Invalid batch_size: {bs}. Must be a positive integer.")
        
        # Check hidden layer sizes
        hls = config.hidden_dims
        if isinstance(hls, (tuple, list)):
            if not all(isinstance(h, int) and h > 0 for h in hls):
                raise ValueError(
                    f"Invalid hidden layer dimensions: {hls}."
                )
        else:
            if not isinstance(hls, int) or hls <= 0:
                raise ValueError(
                    f"Invalid hidden layer dimension: {hls}. Must be a positive integer."
                )
        
        # Check learning rate value
        lr = config.learning_rate
        if not (0 < lr < 1):
            raise ValueError(f"Learning rate too high or invalid: {lr}.")

        # Check weight decay value
        wd = config.weight_decay
        if not (0 <= wd < 1):
            raise ValueError(f"Weight decay invalid: {wd}. Must be >= 0 and < 1.")

    def __initialize_from_scratch(self, run: optuna.Trial, device: torch.device) -> RunState:
        """
        Initializes a new RunState from scratch for a fresh Optuna trial.

        Args:
            run (optuna.Trial): The current run.
            device (torch.device): The target device for model initialization.

        Returns:
            RunState: A fully initialized run state ready for training.
        """

        self.logger.info(f"Initializing state from scratch.")
        
        # Generate hyperparameters
        self.logger.debug(f"Getting run hyperparameters.")
        config: RunConfig = self.setup_run(run)

        # Validate the parameters (optional safety)
        self.logger.debug(f"Checking generated hyperparameters.")
        self.__check_run_config(config)

        # Build checkpoints folder path if does not exist
        checkpoint_dir = Path(self.__output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create RunState
        return RunState(
            checkpoint_path=checkpoint_dir / f"{self.study_id}_{run.number}.pkl",
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dims=config.hidden_dims,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            train_proportion=config.train_proportion,
            batch_size=config.batch_size,
            epochs=config.epochs,
            epoch=0,
            best_model_loss=float("inf"),
            best_model_state_dict={},
            model_state_dict={},
            optimizer_state_dict={},
            wandb_id=f"{self.study_id}_{run.number}",
            wandb_name=f"run_{run.number}_bs{config.batch_size}_hl{'x'.join(map(str, config.hidden_dims))}",
        )

    def __create_checkpoint(self, run: optuna.Trial, checkpoint_path: Path, state: dict) -> None:
        """Atomically saves a checkpoint by writing to a temp file first, then replacing the original.

        Args:
            checkpoint_path (Path): The full path to the .pkl checkpoint file.
            state (dict): The checkpoint state containing training progress.
        """

        # Check that checkpoint path exists
        if not checkpoint_path.parent.exists():
            self.logger.error(f"Specified checkpoint path does not exist: {checkpoint_path.parent}")
            raise

        # Create a temporary file in the same directory
        with tempfile.NamedTemporaryFile(
            dir=checkpoint_path.parent, delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Write to a temporary file first
            with tmp_path.open("wb") as f:
                pickle.dump(state, f)

            # Only overwrite if writing was successful (for atomicity)
            tmp_path.replace(checkpoint_path)
            self.logger.debug(f"Saved checkpoint to '{checkpoint_path}'.")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint at '{checkpoint_path}'.")
            self.logger.error(f"Error: {e}")
            tmp_path.unlink(missing_ok=True)

    # Entry point for running the study
    def run(self, *args) -> None:
        """Runs the Optuna study with the specified objective function and number of trials.

        The study is optimized using the objective function provided, and the number of trials
        is determined by the `n_runs` attribute. A callback is used to limit the number of
        completed trials to `n_runs`.

        Args:
            *args: Additional arguments to pass to the objective function.
        """

        self.study.optimize(
            lambda trial: self.objective(trial, *args),
            n_trials=self.__n_runs,
            callbacks=[
                MaxTrialsCallback(  # This callback will stop the study once the number of trials is reached
                    self.__n_runs, states=(optuna.trial.TrialState.COMPLETE,)
                )
            ],
        )

    def distributed_run(self, n_processes: int, *args) -> None:
        """Starts multiple independent processes running the same study.

        This method initializes and starts a specified number of processes to run the
        `self.run` method concurrently. If `n_processes` is set to 1, it will run the
        `self.run` method in the current process.

        Args:
            n_processes (int): The number of processes to run concurrently.
            *args: Variable length argument list to pass to to the objective function.

        Raises:
            RuntimeError: If the number of processes is less than or equal to 0.
        """

        # Update the number of processes
        self.__n_processes = n_processes

        # Check if n_processes if higher than 0
        if self.__n_processes <= 0:
            raise RuntimeError(
                "Specified number of processes must be equal or greater than 1."
            )

        # Check if n_processes is equal to 1
        if self.__n_processes == 1:
            self.logger.debug("Running in single-process mode.")
            self.run(*args)
            return

        # Initialize processes
        processes: List[multiprocessing.Process] = []
        multiprocessing.set_start_method("spawn", force=True)

        self.logger.info(f"Spawning {self.__n_processes} parallel processes.")

        # Start processes
        for _ in range(self.__n_processes):
            p = multiprocessing.Process(target=self.run, args=args)
            processes.append(p)
            p.start()
            self.logger.debug(f"Process {p.pid} started.")

        # Wait processes to complete
        for p in processes:
            p.join()
            self.logger.debug(f"Process {p.pid} finished.")

    # Main training methods
    def train(self, model: Module, loss_fn: _Loss, train_dl: DataLoader, optimizer: Optimizer, device: torch.device) -> dict:
        # Normalized metrics
        r2_metric = R2Score().to(device)
        mae_metric = MeanAbsoluteError().to(device)
        mse_metric = MeanSquaredError().to(device)

        # Initialize loss variables
        total_loss = 0.0
        total_samples = 0

        # Per-output metrics
        per_output_mae = {
            name: MeanAbsoluteError().to(device)
            for name in self.__output_mapping
        }
        per_output_mse = {
            name: MeanSquaredError().to(device)
            for name in self.__output_mapping
        }

        model.train()

        for inputs, targets in train_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
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

            # Per-variable denormalized metrics
            if outputs_real.shape[1] != len(self.__output_mapping):
                raise ValueError(
                    f"Mismatch between number of outputs ({outputs_real.shape[1]}) "
                    f"and output_mapping ({len(self.__output_mapping)} entries)."
                )
            for i, name in enumerate(self.__output_mapping):
                per_output_mae[name].update(outputs_real[:, i], targets_real[:, i])
                per_output_mse[name].update(outputs_real[:, i], targets_real[:, i])

        # Final aggregation
        metrics = {
            "train/loss": total_loss / total_samples,
            "train/mse": mse_metric.compute().item(),
            "train/mae": mae_metric.compute().item(),
            "train/r2": r2_metric.compute().item(),
        }

        for name in self.__output_mapping:
            metrics[f"train/{name}_mae"] = per_output_mae[name].compute().item()
            metrics[f"train/{name}_mse"] = per_output_mse[name].compute().item()

        return metrics

    def evaluate(self, model: torch.nn.Module, loss_fn: _Loss, test_dl: DataLoader, device: torch.device) -> dict:
        # Normalized metrics
        r2_metric = R2Score().to(device)
        mae_metric = MeanAbsoluteError().to(device)
        mse_metric = MeanSquaredError().to(device)

        # Initialize loss variables
        total_loss = 0.0
        total_samples = 0

        # Per-output metrics
        per_output_mae = {
            name: MeanAbsoluteError().to(device)
            for name in self.__output_mapping
        }
        per_output_mse = {
            name: MeanSquaredError().to(device)
            for name in self.__output_mapping
        }

        model.eval()
        with torch.no_grad():
            for inputs, targets in test_dl:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Compute loss
                loss = loss_fn(outputs, targets)

                # Accumulate weighted loss
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Update normalized metrics
                r2_metric.update(outputs, targets)
                mae_metric.update(outputs, targets)
                mse_metric.update(outputs, targets)

                # Denormalize outputs and targets
                outputs_real = self.denormalize(outputs)
                targets_real = self.denormalize(targets)

                # Per-variable denormalized metrics
                if outputs_real.shape[1] != len(self.__output_mapping):
                    raise ValueError(
                        f"Mismatch between number of outputs ({outputs_real.shape[1]}) "
                        f"and output_mapping ({len(self.__output_mapping)} entries)."
                    )
                for i, name in enumerate(self.__output_mapping):
                    per_output_mae[name].update(outputs_real[:, i], targets_real[:, i])
                    per_output_mse[name].update(outputs_real[:, i], targets_real[:, i])

        # Final aggregation
        metrics = {
            "evaluate/loss": total_loss / total_samples,
            "evaluate/mse": mse_metric.compute().item(),
            "evaluate/mae": mae_metric.compute().item(),
            "evaluate/r2": r2_metric.compute().item(),
        }

        for name in self.__output_mapping:
            metrics[f"evaluate/{name}_mae"] = per_output_mae[name].compute().item()
            metrics[f"evaluate/{name}_mse"] = per_output_mse[name].compute().item()

        return metrics

    def objective(
        self, 
        run: optuna.Trial,
        data_path: str
    ) -> float:
        """Objective function to optimize. To be implemented in child classes."""
        
        # Initialize the run logger
        self.__init_run_logger(run)

        # Setup PyTorch device
        if torch.cuda.is_available():
            device_index = run.number % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_index}")
            self.logger.info(f"Using '{torch.cuda.device_count()}' CUDA devices.")
        else:
            device = torch.device("cpu")
            self.logger.info(f"Using CPU.")

        # Resume or initialize the run state
        try:
            state = self.__try_resume_from_checkpoint(run, device)
            self.logger.info(f"Resuming run from checkpoint at epoch {state.epoch}.")
        except NoCheckpointAvailable:
            # Initialize the state for the  run, from scratch
            self.logger.debug(f"No checkpoint found for run {run.number}.")
            state = self.__initialize_from_scratch(run, device)

            # Storing run information in Optuna database
            run.set_user_attr("checkpoint_path", str(state.checkpoint_path))
            run.set_user_attr("batch_size", state.batch_size)
            run.set_user_attr("learning_rate", state.learning_rate)
            run.set_user_attr("weight_decay", state.weight_decay)
            run.set_user_attr("train_proportion", state.train_proportion)
            run.set_user_attr("epochs", state.epochs)
            run.set_user_attr("input_dim", state.input_dim)
            run.set_user_attr("output_dim", state.output_dim)
            run.set_user_attr("hidden_dims", state.hidden_dims)
            run.set_user_attr("wandb_id", state.wandb_id)
            run.set_user_attr("wandb_name", state.wandb_name) 
        except (KeyError, RuntimeError, ValueError) as e:
            # Crash early to avoid proceeding with invalid state
            self.logger.error(f"Checkpoint is corrupted or incomplete: {e}")
            raise

        # Initialize WandB run | REFERENCE FOR DISTRIBUTED: https://docs.wandb.ai/support/multiprocessing_eg_distributed_training/
        wandb_run = wandb.init(
            id=state.wandb_id,
            project=self.study_id,
            name=state.wandb_name,
            group="neuraltrain",
            config={
                "input_dim": state.input_dim,
                "output_dim": state.output_dim,
                "hidden_dims": state.hidden_dims,
                "batch_size": state.batch_size,
                "train_portion": state.train_proportion,
                "evaluation_portion": round(1 - state.train_proportion),
                "learning_rate": state.learning_rate,
                "weight_decay": state.weight_decay,
            },
            reinit=True,
            dir=self.__output_dir,
        )

        input_dim = state.input_dim
        hidden_dims = state.hidden_dims
        output_dim = state.output_dim

        self.logger.debug(f"Initializing model with input_dim: {input_dim}, hidden_dims: {hidden_dims}, output_dim: {output_dim}.")
        # Instantiate model and load state if available
        model = self.__model_class(
            input_dim, 
            hidden_dims, 
            output_dim
        ).to(device)
        if state.model_state_dict:
            model.load_state_dict(state.model_state_dict)

        self.logger.debug(f"Initializing optimizer with learning_rate: {state.learning_rate}, weight_decay: {state.weight_decay}.")
        # Instantiate optimizer and load state if available
        optimizer: Optimizer = self.__optimizer_class(
            model.parameters(),
            lr=state.learning_rate,
            weight_decay=state.weight_decay,
        )
        if state.optimizer_state_dict:
            optimizer.load_state_dict(state.optimizer_state_dict)

        # Load dataset and get dataloaders
        df = pd.read_csv(data_path)
        train_dl, test_dl = TimeSeriesDataset.get_dataloaders(
            df,
            input_col_filter=self.input_col_filter,
            target_col_filter=self.target_col_filter,
            train_proportion=state.train_proportion,
            batch_size=state.batch_size,
        )

        input_cols = list(filter(self.input_col_filter, df.columns))
        target_cols = list(filter(self.target_col_filter, df.columns))
        self.logger.debug(f"Dataset input columns: {input_cols}")
        self.logger.debug(f"Dataset target columns: {target_cols}")
        self.logger.debug(f"Dataset loaded {len(df)} entries.")

        loss_fn = self.loss_function()

        # NN training loop
        for epoch in range(state.epoch, state.epochs, 1):
            # Train the model for one epoch
            train_metrics = self.train(
                model, loss_fn, train_dl, optimizer, device
            )
            eval_metrics = self.evaluate(model, loss_fn, test_dl, device)

            # Get the evaluation loss
            eval_loss = eval_metrics["evaluate/loss"]

            # Save the best model
            if eval_loss < state.best_model_loss:
                state = replace(
                    state,
                    best_model_loss=eval_loss,
                    best_model_state_dict=model.state_dict()
                )

            # Log information
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    **train_metrics,
                    **eval_metrics,
                },
                step=epoch + 1,
            )

            # Save checkpoint every 50 epochs
            if epoch % 50 == 0:
                state = replace(
                    state,
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict()
                )
                self.logger.info(
                    f"Saving checkpoint at epoch index {epoch}."
                )
                self.__create_checkpoint(run, state.checkpoint_path, state.to_checkpoint_dict())

        # Store tracked best model
        model_path = (
            Path(self.__output_dir) / f"{state.wandb_name}.pth"
        )
        torch.save(state.best_model_state_dict, model_path)

        self.logger.info("=" * 70)
        self.logger.info(f"✅ Run {run.number} completed")
        self.logger.info(f" Best model saved at: {model_path}")
        self.logger.info(f" Best validation loss: {state.best_model_loss:.6f}")
        self.logger.info(f" Best model hyperparameters:")
        self.logger.info(f"   • Input Dimension: {state.input_dim}")
        self.logger.info(f"   • Output Dimension: {state.output_dim}")
        self.logger.info(f"   • Hidden Layers Sizes: {state.hidden_dims}")
        self.logger.info(f"   • Batch Size: {state.batch_size}")
        self.logger.info(f"   • Learning Rate: {state.learning_rate}")
        self.logger.info(f"   • Weight Decay: {state.weight_decay}")
        self.logger.info(f"   • Train Proportion: {state.train_proportion}")
        self.logger.info(f"   • Epochs: {state.epochs}")
        self.logger.info("=" * 70)

        wandb_run.finish()
        return state.best_model_loss

    # To be implemented in child classes
    @abstractmethod
    def setup_run(self, run: optuna.Trial) -> RunConfig:
        """
        Generate and return a RunConfig object containing all resolved hyperparameters for a given Optuna trial.

        Each argument can be one of:
        - A fixed value (e.g., `epochs=2000`) which will be used as-is.
        - A list of options (e.g., `batch_size=[128, 256, 512]`), where one will be selected using `run.suggest_categorical`.
        - A list of two values (e.g., `learning_rate=[1e-4, 1e-2]`), where a value will be sampled using `run.suggest_float`.

        Args:
            run (optuna.Trial): The Optuna trial object used to suggest values.

        Returns:
            RunConfig: A dataclass instance containing all resolved and ready-to-use hyperparameters:
                - hidden_dims (List[int]): List of hidden layer sizes. 
                  The length of the list is determined by the number of hidden layers in the specified model.
                - batch_size (int): Training batch size.
                - epochs (int): Number of training epochs.
                - train_proportion (float): Proportion of dataset used for training (0 < value < 1).
                - learning_rate (float): Learning rate for the optimizer.
                - weight_decay (float): L2 regularization coefficient.
                - input_dim (int): Input feature dimension (copied from argument).
                - output_dim (int): Output feature dimension (copied from argument).
                - wandb_id (str): Unique identifier for the W&B run.
                - wandb_name (str): Human-readable name for the W&B run.
                - checkpoint_path (str): Path to save the run checkpoint.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")

    @abstractmethod
    def loss_function(self) -> _Loss:
        """
        Returns the loss function to be used in training and validation.

        This method must be implemented by subclasses to return an instance of a PyTorch
        loss module suitable for regression tasks.

        Examples of loss functions for regression in PyTorch include:
            - torch.nn.MSELoss()           # Mean Squared Error Loss
            - torch.nn.L1Loss()            # Mean Absolute Error (L1) Loss
            - torch.nn.SmoothL1Loss()      # Huber loss, less sensitive to outliers
            - torch.nn.HuberLoss()         # Same as SmoothL1Loss, more explicit name (PyTorch ≥1.10)
            - torch.nn.PoissonNLLLoss()    # Poisson Negative Log-Likelihood Loss (for count data)

        Implementation example:
            return torch.nn.MSELoss()  

        Returns:
            torch.nn.modules.loss._Loss: A loss function instance compatible with PyTorch.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")

    @abstractmethod
    def input_col_filter(self, col: str) -> bool:
        """Filter function to select input columns based on a condition."""
        raise NotImplementedError("This method should be overridden in subclasses.")

    @abstractmethod
    def target_col_filter(self, col: str) -> bool:
        """Filter function to select target columns based on a condition."""
        raise NotImplementedError("This method should be overridden in subclasses.")
    
    @abstractmethod
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize the data to its original scale."""
        raise NotImplementedError("This method should be overridden in subclasses.")

    # Getter methods for private attributes
    @property
    def db_url(self) -> str:
        return self.__db_url
    
    @property
    def study_id(self) -> str:
        return self.__study_id
    
    @property
    def n_runs(self) -> int:
        return self.__n_runs
    
    @property
    def output_dir(self) -> str:
        return str(self.__output_dir.absolute())
    
    @property
    def model_class(self) -> Module:
        return self.__model_class
    
    @property
    def optimizer_class(self) -> Optimizer:
        return self.__optimizer_class
    
    @property
    def debug_mode(self) -> bool:
        return self.__debug_mode
    
    @property
    def n_processes(self) -> int:
        return self.__n_processes
