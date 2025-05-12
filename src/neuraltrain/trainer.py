import optuna
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
from typing import Callable


class NeuralTrainerBase(ABC):
    """Abstract class to implement a Neural Network training levaraging Optuna for optimization and paralelism."""

    def __init__(
        self,
        db_url: str,
        study_name: str,
        n_trials: int,
        output_dir: Path,
        model,
        optimizer,
        sampler=None,
        pruner=None,
        debug_mode=False,
    ):
        # Set relevant global attributes to the entire study
        self.__db_url = db_url
        self.__study_name = study_name
        self.__n_trials = n_trials
        self.__output_dir = output_dir
        self.__debug_mode = debug_mode
        self.__model_class: torch.nn.Module = model
        self.__optimizer_class = optimizer

        # Setting up the logger
        self.logger = logging.getLogger("NeuralTrainBase")

        # Setting the log level based on debug mode
        if self.__debug_mode:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

        # Setting up the formatter for structured logging
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.debug(
            f"Initializing study '{self.__study_name}' from database '{self.__db_url}'."
        )

        # Manually check for COMPLETE trials and subtract them from n_trials
        all_trials = self.__get_all_trials()
        if all_trials:
            completed_trials = [
                trial for trial in all_trials if trial["state"] == "COMPLETE"
            ]
            self.logger.debug(
                f"Found {len(completed_trials)} COMPLETED trials in study '{self.__study_name}'."
            )

            failed_trials = [
                trial for trial in all_trials if trial["state"] in ["FAIL", "RUNNING"]
            ]
            self.logger.debug(
                f"Found {len(failed_trials)} FAIL/RUNNING trials in study '{self.__study_name}'."
            )

            # Subtract completed trials from total trials
            self.__n_trials -= len(completed_trials)

            # Re-enqueue failed trials
            if failed_trials:
                self.logger.warning(
                    f"Re-enqueuing {len(failed_trials)} failed trial(s) in study '{self.__study_name}'."
                )
                self.__reenqueue_failed_trials(
                    [trial["trial_id"] for trial in failed_trials]
                )

        # Create a new study or load an existing one
        self.study = optuna.create_study(
            storage=self.__db_url,
            study_name=self.__study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        self.logger.debug(f"Study '{self.__study_name}' initialized successfully.")

    def run(self, *args) -> None:
        """Runs the Optuna study with the specified objective function and number of trials.

        The study is optimized using the objective function provided, and the number of trials
        is determined by the `n_trials` attribute. A callback is used to limit the number of
        completed trials to `n_trials`.

        Args:
            *args: Additional arguments to pass to the objective function.
        """

        self.study.optimize(
            lambda trial: self.objective(trial, *args),
            n_trials=self.__n_trials,
            callbacks=[
                MaxTrialsCallback(  # This callback will stop the study once the number of trials is reached
                    self.__n_trials, states=(optuna.trial.TrialState.COMPLETE,)
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

        # Check if n_processes if higher than 0
        if n_processes <= 0:
            raise RuntimeError(
                "Specified number of processes must be equal or greater than 1."
            )

        # Check if n_processes is equal to 1
        if n_processes == 1:
            self.logger.debug("Running in single-process mode.")
            self.run(*args)
            return

        # Initialize processes
        processes: List[multiprocessing.Process] = []
        multiprocessing.set_start_method("spawn", force=True)

        self.logger.info(f"Spawning {n_processes} parallel processes.")

        # Start processes
        for _ in range(n_processes):
            p = multiprocessing.Process(target=self.run, args=args)
            processes.append(p)
            p.start()
            self.logger.debug(f"Process {p.pid} started.")

        # Wait processes to complete
        for p in processes:
            p.join()
            self.logger.debug(f"Process {p.pid} finished.")

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
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

    def create_checkpoint(self, checkpoint_path: Path, state: dict) -> None:
        """Atomically saves a checkpoint by writing to a temp file first, then replacing the original.

        Args:
            checkpoint_path (Path): The full path to the .pkl checkpoint file.
            state (dict): The checkpoint state containing training progress.
        """

        # Ensure the parent directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

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
            self.logger.info(f"Saved checkpoint to '{checkpoint_path}'.")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint at '{checkpoint_path}'.")
            self.logger.error(f"Error: {e}")
            tmp_path.unlink(missing_ok=True)

    def __get_all_trials(self) -> List[Dict[str, Any]]:
        """Retrieves all trials from the Optuna SQLite database for the current study."""

        # Extract SQLite database path from the URL
        db_path = self.__db_url.replace("sqlite:///", "")

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
            "SELECT study_id FROM studies WHERE study_name = ?", (self.study_name,)
        ).fetchone()

        # Check if the study exists, if not return an empty list
        if result is None:
            conn.close()
            self.logger.warning(
                f"Study '{self.__study_name}' not found in database. Expected if it is the first time."
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
            f"Re-enqueued {affected_rows} failed trial(s) in study '{self.__study_name}'."
        )

    def __check_trial_params(self, params: Dict[str, Any]) -> None:
        """Raise descriptive errors for invalid hyperparameter values."""
        tp = params["train_proportion"]
        if not 0.0 < tp < 1.0:
            raise ValueError(
                f"Invalid train_proportion: {tp}. Must be between 0 and 1 (exclusive)."
            )

        bs = params["batch_size"]
        if not isinstance(bs, int) or bs <= 0:
            raise ValueError(f"Invalid batch_size: {bs}. Must be a positive integer.")

        hl1, hl2 = params["hidden_dims"]
        if not all(isinstance(h, int) and h > 0 for h in (hl1, hl2)):
            raise ValueError(
                f"Invalid hidden layer dimensions: {params['hidden_dims']}."
            )

        lr = params["learning_rate"]
        if not (0 < lr < 1):
            raise ValueError(f"Learning rate too high or invalid: {lr}.")

        wd = params["weight_decay"]
        if not (0 <= wd < 1):
            raise ValueError(f"Weight decay invalid: {wd}. Must be >= 0 and < 1.")

    def loss_function(self) -> torch.nn.Module:
        """
        Returns the loss function to be used in training.

        By default, this uses Mean Squared Error loss (MSE), which is
        common for regression tasks. Override this method in subclasses
        to customize the loss function.

        Returns:
            torch.nn.Module: The loss function instance.
        """
        return torch.nn.MSELoss()

    def train(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        train_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        # Normalized metrics
        r2_metric = R2Score().to(device)
        mae_metric = MeanAbsoluteError().to(device)
        mse_metric = MeanSquaredError().to(device)

        # New: Denormalized metrics
        mae_real_metric = MeanAbsoluteError().to(device)
        mse_real_metric = MeanSquaredError().to(device)

        model.train()
        for inputs, targets in train_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalized updates
            r2_metric.update(outputs, targets)
            mae_metric.update(outputs, targets)
            mse_metric.update(outputs, targets)

            # Desnormalize (assumindo radiação entre 0 e 1000)
            outputs_real = self.denormalize(outputs)
            targets_real = self.denormalize(targets)

            # Real updates
            mae_real_metric.update(outputs_real, targets_real)
            mse_real_metric.update(outputs_real, targets_real)

        return {
            "train/mse": mse_metric.compute().item(),
            "train/mae": mae_metric.compute().item(),
            "train/r2": r2_metric.compute().item(),
            "train/mse_real": mse_real_metric.compute().item(),
            "train/mae_real": mae_real_metric.compute().item(),
        }

    def evaluate(
        self, model: torch.nn.Module, test_dl: DataLoader, device: torch.device
    ):
        # Normalized metrics
        r2_metric = R2Score().to(device)
        mae_metric = MeanAbsoluteError().to(device)
        mse_metric = MeanSquaredError().to(device)

        # New: Denormalized metrics
        mae_real_metric = MeanAbsoluteError().to(device)
        mse_real_metric = MeanSquaredError().to(device)

        model.eval()
        with torch.no_grad():
            for inputs, targets in test_dl:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Update normalized metrics
                r2_metric.update(outputs, targets)
                mae_metric.update(outputs, targets)
                mse_metric.update(outputs, targets)

                # Desnormalize (assumindo que é radiação solar normalizada de 0 a 1000)
                outputs_real = self.denormalize(outputs)
                targets_real = self.denormalize(targets)

                # Update real metrics
                mae_real_metric.update(outputs_real, targets_real)
                mse_real_metric.update(outputs_real, targets_real)

        return {
            "evaluate/mse": mse_metric.compute().item(),
            "evaluate/mae": mae_metric.compute().item(),
            "evaluate/r2": r2_metric.compute().item(),
            "evaluate/mse_real": mse_real_metric.compute().item(),
            "evaluate/mae_real": mae_real_metric.compute().item(),
        }

    def objective(
        self,
        trial: optuna.Trial,
        hidden_dims: list,
        input_dim: int,
        output_dim: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        df_path: str,
        train_proportion: float,
        batch_sizes: list,
        output_dir: str,
        study_name: str,
    ) -> float:
        """Objective function to optimize. To be implemented in child classes."""

        # Setup PyTorch device
        if torch.cuda.is_available():
            device_index = trial.number % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_index}")
            self.logger.info(f"Using CUDA device: {device} - {torch.cuda.get_device_name(device_index)}")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")

        # Look for existing trials and checkpoints
        if trial.user_attrs:
            try:
                # Load the relevant checkpoint attributes
                checkpoint_path = Path(trial.user_attrs["checkpoint_path"])
                hidden_dim_1, hidden_dim_2 = trial.user_attrs["hidden_dims"]
                batch_size = trial.user_attrs["batch_size"]
                learning_rate = trial.user_attrs["learning_rate"]
                weight_decay = trial.user_attrs["weight_decay"]
                input_dim = trial.user_attrs["input_dim"]
                output_dim = trial.user_attrs["output_dim"]
                wandb_id = trial.user_attrs["wandb_id"]
                wandb_name = trial.user_attrs["wandb_name"]
                epochs = trial.user_attrs["epochs"]
                train_proportion = trial.user_attrs["train_proportion"]

                self.logger.info(
                    f"Loading checkpoint from: {checkpoint_path} for trial {trial.number} (hidden_dims={hidden_dims})..."
                )

                # Load the checkpoint
                checkpoint = self.load_checkpoint(checkpoint_path)

                # Check if the checkpoint is valid and has all the fields
                if checkpoint:
                    # Initialize the model and optimizer
                    model: torch.nn.Module = self.__model_class(
                        input_dim, hidden_dim_1, hidden_dim_2, output_dim
                    ).to(device)

                    optimizer: torch.optim.Optimizer = self.__optimizer_class(
                        model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay,
                    )

                    # Load the model and optimizer states from checkpoint
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                    # Load progress variables
                    starting_epoch = checkpoint["epoch"]
                    best_model_loss = checkpoint["best_model_loss"]
                    best_model_state = checkpoint["best_model_state"]
                    best_model_parameters = checkpoint["best_model_parameters"]

                    self.logger.info(
                        f"Checkpoint loaded. Resuming training from epoch {starting_epoch}."
                    )
                else:
                    self.logger.error(
                        f"Missing or invalid checkpoint for trial {trial.number}. Aborting with an error."
                    )
                    raise RuntimeError(
                        f"Checkpoint is empty or invalid for trial {trial.number}."
                    )

            except Exception as e:
                self.logger.error(f"Failed to resume trial {trial.number}: {e}")
                raise RuntimeError(
                    f"Failed to load checkpoint for trial {trial.number}"
                ) from e
        else:
            # No valid checkpoint, setup the trial
            self.logger.info(f"🔄 Setting up trial {trial.number} from scratch.")
            trial_info = self.setup_trial(
                trial,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                batch_sizes=batch_sizes,
                epochs=epochs,
                train_proportion=train_proportion,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )

            # Check if the trial_info is valid
            self.__check_trial_params(trial_info)

            # Unpack everything from trial_info
            hidden_dim_1, hidden_dim_2 = trial_info["hidden_dims"]
            batch_size = trial_info["batch_size"]
            learning_rate = trial_info["learning_rate"]
            weight_decay = trial_info["weight_decay"]
            input_dim = trial_info["input_dim"]
            output_dim = trial_info["output_dim"]
            epochs = trial_info["epochs"]
            train_proportion = trial_info["train_proportion"]
            checkpoint_path = Path(trial_info["checkpoint_path"])
            wandb_id = trial_info["wandb_id"]
            wandb_name = trial_info["wandb_name"]

            model = self.__model_class(
                input_dim, hidden_dim_1, hidden_dim_2, output_dim
            ).to(device)
            optimizer = self.__optimizer_class(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

            starting_epoch = 0
            best_model_loss = float("inf")
            best_model_state = None
            best_model_parameters = []

            model: torch.nn.Module = self.__model_class(
                input_dim, hidden_dim_1, hidden_dim_2, output_dim
            ).to(device)

            optimizer: torch.optim.Optimizer = self.__optimizer_class(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            # Storing trial information
            trial.set_user_attr("checkpoint_path", str(checkpoint_path))
            trial.set_user_attr("hidden_dims", [hidden_dim_1, hidden_dim_2])
            trial.set_user_attr("batch_size", batch_size)
            trial.set_user_attr("learning_rate", learning_rate)
            trial.set_user_attr("weight_decay", weight_decay)
            trial.set_user_attr("input_dim", input_dim)
            trial.set_user_attr("output_dim", output_dim)
            trial.set_user_attr("wandb_id", wandb_id)
            trial.set_user_attr("wandb_name", wandb_name)
            trial.set_user_attr("epochs", epochs)
            trial.set_user_attr("train_proportion", train_proportion)

        # Load and prepare the dataloaders
        df = pd.read_csv(df_path)
        train_dl, test_dl = TimeSeriesDataset.get_dataloaders(
            df,
            input_col_filter=self.input_col_filter,
            target_col_filter=self.target_col_filter,
            train_proportion=train_proportion,
            batch_size=batch_size,
        )

        # Initialize Weights and Biases run REFERENCE FOR DISTRIBUTED: https://docs.wandb.ai/support/multiprocessing_eg_distributed_training/
        run = wandb.init(
            id=wandb_id,
            project=study_name,
            name=wandb_name,
            group="optuna-multiprocessing",
            config={
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim_layer_1": hidden_dim_1,
                "hidden_dim_layer_2": hidden_dim_2,
                "batch_size": batch_size,
                "train_portion": train_proportion,
                "evaluation_portion": 1 - train_proportion,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
            reinit=True,
            dir=output_dir,
        )

        # Start training loop
        for epoch in range(starting_epoch, epochs, 1):
            # Train and evaluate the model
            train_metrics = self.train(
                model, self.loss_function(), train_dl, optimizer, device
            )
            eval_metrics = self.evaluate(model, test_dl, device)

            # Get the evaluation loss
            eval_loss = eval_metrics["evaluate/mse"]

            # Save the best model
            if eval_loss < best_model_loss:
                best_model_loss = eval_loss
                best_model_state = model.state_dict()
                best_model_parameters = [hidden_dim_1, hidden_dim_2, batch_size, epoch]

            # Log information
            run.log(
                {
                    "epoch": epoch + 1,
                    **train_metrics,
                    **eval_metrics,
                },
                step=epoch + 1,
            )

            # Save checkpoint every 50 epochs
            if epoch % 50 == 0:
                state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_model_loss": best_model_loss,
                    "best_model_state": best_model_state,
                    "best_model_parameters": best_model_parameters,
                }
                self.create_checkpoint(checkpoint_path, state)

        # Store tracked best model
        hidden_dim_1, hidden_dim_2, batch_size, epoch = best_model_parameters
        model_path = (
            Path(output_dir) / f"model_1hl{hidden_dim_1}_2hl{hidden_dim_2}_bs{batch_size}_e{epoch+1}.pth"
        )
        torch.save(best_model_state, model_path)

        run.finish()
        return best_model_loss

    @property
    def output_dir(self) -> Path:
        return self.__output_dir

    @property
    def study_name(self) -> str:
        return self.__study_name

    @property
    def study_id(self) -> int:
        return self.study._study_id

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

    @abstractmethod
    def setup_trial(
        self,
        trial: optuna.Trial,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] | int,
        batch_size: List[int] | int,
        epochs: List[int] | int,
        train_proportion: List[float] | float,
        learning_rate: List[float] | float,
        weight_decay: List[float] | float,
    ) -> Dict[str, Any]:
        """
        Generate the hyperparameters for the given Optuna trial.

        Each argument can be one of:
        - A **fixed value** (e.g., `epochs=2000`) which will be used as-is.
        - A **list of options** (e.g., `batch_size=[128, 256, 512]`), where one will be selected using `trial.suggest_categorical`.
        - A **range** list of two values (e.g., `learning_rate=[1e-4, 1e-2]`), where a value will be sampled using `trial.suggest_float`.

        Example implementation:
            - `hidden_dims = [64, 256]` → sampled integer from range using `trial.suggest_int`
            - `batch_size = [256, 512, 1024]` → picked from options using `trial.suggest_categorical`
            - `epochs = 2000` → used as a fixed value
            - `train_proportion = [0.6, 0.9]` → sampled float from range using `trial.suggest_float`
            - `learning_rate = [1e-4, 1e-2]` → sampled from log scale using `trial.suggest_float(log=True)`
            - `weight_decay = 5e-6` → used as fixed value

        Returns:
            Dict[str, Any]: Dictionary of resolved parameters to use in the trial, including:
                - hidden_dims (List[int])
                - batch_size (int)
                - epochs (int)
                - train_proportion (float)
                - learning_rate (float)
                - weight_decay (float)
                - input_dim (int)
                - output_dim (int)
                - wandb_id (str)
                - wandb_name (str)
        """
        raise NotImplementedError("This method should be overridden in subclasses.")
