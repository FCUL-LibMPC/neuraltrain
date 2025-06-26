import sys
import torch
import wandb
import pickle
import optuna
import sqlite3
import logging
import tempfile
import multiprocessing
from typing import List
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from optuna.study import MaxTrialsCallback
from typing import List, Dict, Any, Tuple, Type


# Type aliases for search space and configurations
HyperParameter = Tuple[Tuple[int, int], int, int, float, float, float]
SearchSpace = List[HyperParameter]
ModelType = Type[torch.nn.Module]
OptimizerType = Type[torch.optim.Optimizer]

# Dataclass configurations for study and model
@dataclass(frozen=True)
class StudyConfig:
    sqlite_url: str
    """SQLite3 database URL for storing study results."""
    study_id: str
    """Unique identifier for the study, used to group results and to identify logs in Weights&Biases."""
    search_space: SearchSpace
    """Cartesian product of hyperparameter combinations.
    Each tuple is:
    ((hidden_dim_1, hidden_dim_2, ...), batch_size, epochs, learning_rate, weight_decay, train_proportion)
    """
    output_dir: Path
    """Path to the directory in which the results and local logs will be stored."""
    debug_mode: bool = False
    """If True, enables debug logging and additional checks."""

@dataclass(frozen=True)
class ModelConfig:
    model_class: ModelType
    """Class of the ANN model to be trained. Must be a subclass of `torch.nn.Module`."""
    optimizer_class: OptimizerType
    """Class of the optimizer to be used for training. Must be a subclass of `torch.optim.Optimizer`."""
    input_dim: int
    """Dimension of the input laye, equivalent to the size of the model input array."""
    output_dim: int
    """Dimension of the output layer, equivalent to the size of the model output array."""

@dataclass(frozen=True)
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

    def to_database_dict(self) -> Dict[str, Any]:
        """
        Extracts the run-relevant training information from the RunState to store in the Optuna database.
        Remaining information is not relevant for the database as it is stored in the checkpoint.

        ### Returns:
        - dict: A dictionary containing run-relevant training information for the Optuna database.
        """
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "train_proportion": self.train_proportion,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "wandb_id": self.wandb_id,
            "wandb_name": self.wandb_name
        }

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """
        Extracts the checkpoint-relevant training state from the RunState to store as pickled dictionary.
        Remaining information is not relevant for checkpointing as it is stored in the database.

        ### Returns:
        - dict: A dictionary containing runtime training state for checkpointing.
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

class NeuralNetworkTrainFramework(ABC):
    """
    Abstract base class that provides a structured framework for training neural network models with support for
    reproducible experiment management, including checkpointing, resuming trials, logging (e.g., via Weights & Biases),
    and parallel execution using Optuna.

    This class is not responsible for implementing the actual model training or evaluation logic. Instead, it defines
    the infrastructure and lifecycle for a training experiment, while delegating the training and evaluation routines
    to subclasses via abstract methods.
    """

    def __init__(self, study_config: StudyConfig, model_config: ModelConfig):
        """
        Initializes the training framework with the provided study and model configurations.

        ### Args:
        - study_config (StudyConfig): Configuration for the study, including database URL, study ID, number of runs, and output directory.
        - model_config (ModelConfig): Configuration for the model, including model class, optimizer class, input dimension, and output dimension.
        """

        # Study information
        self._sqlite_url = study_config.sqlite_url
        self._study_id = study_config.study_id
        self._search_space = study_config.search_space
        self._num_runs = len(self._search_space)
        self._output_dir = study_config.output_dir / f"{self._study_id}"
        self._num_processes = 1  # Default to single process; Can be overridden later at self.distributed_run()
        self._debug_mode = study_config.debug_mode

        # Model information
        self._model_class = model_config.model_class
        self._optimizer_class = model_config.optimizer_class
        self._input_dim = model_config.input_dim
        self._output_dim = model_config.output_dim

        # Make sure the output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Setup the logger
        self.__init_logger()

        # Check and recover from previously failed/incomplete study
        all_runs = self.__get_all_runs()
        if all_runs:
            # Get completed runs
            completed_runs = [
                run for run in all_runs if run["state"] == "COMPLETE"
            ]
            self.logger.debug(
                f"Found {len(completed_runs)} COMPLETED run(s) in study '{self._study_id}'."
            )

            # Get failed runs
            failed_runs = [
                run for run in all_runs if run["state"] in ["FAIL", "RUNNING"]
            ]
            self.logger.debug(
                f"Found {len(failed_runs)} FAIL/RUNNING run(s) in study '{self._study_id}'."
            )

            # Recover study from failed state; Subtract completed runs from n_runs
            self._num_runs -= len(completed_runs)

            # Re-enqueue failed runs
            if failed_runs:
                self.logger.warning(
                    f"Re-enqueuing {len(failed_runs)} failed run(s) in study '{self._study_id}'."
                )
                self.__reenqueue_failed_trials(
                    [run["trial_id"] for run in failed_runs]
                )

        # Create a new study or load an existing one
        self.study = optuna.create_study(
            storage=self._sqlite_url,
            study_name=self._study_id,
            direction="minimize",
            load_if_exists=True,
        )

        if self._num_runs == 0:
            return self.logger.warning(
                f"All runs in study '{self._study_id}' are already completed. No new runs will be started."
            )

        self.logger.info(f"Study '{self._study_id}' initialized successfully.")

    # Logger related initialization methods
    def __init_logger(self) -> None:
        """Initializes the logger for the training framework."""
        # Setting up the logger
        logger = logging.getLogger(self.__class__.__name__)
        level = logging.DEBUG if self._debug_mode else logging.INFO
        logger.setLevel(level)

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
        logger.info(f"Logger initialized for {self.__class__.__name__}")

    def __init_run_logger(self, run: optuna.Trial) -> None:
        """
        Initializes a per-run logger with the run number included in the log format.

        Args:
            run_number (int): The Optuna trial number for this run.
        """
        logger_name = f"{self.__class__.__name__}.Run{run.number}"
        logger = logging.getLogger(logger_name)

        level = logging.DEBUG if self._debug_mode else logging.INFO
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
        logger.info(f"Logger initialized for '{logger_name}'.")

    # Study initialization and recovery methods
    def __reenqueue_failed_trials(self, trial_ids: List[int]) -> None:
        """Manually re-enqueues failed trials back to WAITING state in the Optuna SQLite database.

        Args:
            trial_ids (List[int]): A list of trial IDs to re-enqueue.
        """

        if not trial_ids:
            return

        # Extract the database path from the URL
        db_path = self._sqlite_url.replace("sqlite:///", "")

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
            f"Re-enqueued {affected_rows} failed trial(s) in study '{self._study_id}'."
        )

    def __get_all_runs(self) -> List[Dict[str, Any]]:
        """Retrieves all trials from the Optuna SQLite database for the current study."""

        # Extract SQLite database path from the url
        db_path = self._sqlite_url.removeprefix("sqlite:///")

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
            "SELECT study_id FROM studies WHERE study_name = ?", (self._study_id,)
        ).fetchone()

        # Check if the study exists
        if result is None:
            conn.close()
            self.logger.warning(
                f"Study '{self._study_id}' not found in database. Expected if it is the first time."
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
        if self._debug_mode and trials_list:
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

    def __check_run_config(self, hl, bs, ep, lr, wd, t_proportion) -> HyperParameter:
        """Raise descriptive error for invalid values ir configuration."""

        # Check train proportion
        if not 0.0 < t_proportion < 1.0:
            raise ValueError(
                f"Invalid train_proportion: {t_proportion}. Must be between 0 and 1 (exclusive)."
            )
        
        # Check batchsize value
        if not isinstance(bs, int) or bs <= 0:
            raise ValueError(f"Invalid batch_size: {bs}. Must be a positive integer.")
        
        # Check hidden layer sizes
        if isinstance(hl, (tuple, list)):
            if not all(isinstance(h, int) and h > 0 for h in hl):
                raise ValueError(
                    f"Invalid hidden layer dimensions: {hl}."
                )
        else:
            if not isinstance(hl, int) or hl <= 0:
                raise ValueError(
                    f"Invalid hidden layer dimension: {hl}. Must be a positive integer."
                )
        
        # Check learning rate value
        if not (0 < lr < 1):
            raise ValueError(f"Learning rate too high or invalid: {lr}.")

        # Check weight decay value
        if not (0 <= wd < 1):
            raise ValueError(f"Weight decay invalid: {wd}. Must be >= 0 and < 1.")
        
        # Check epochs value
        if not isinstance(ep, int) or ep <= 0:
            raise ValueError(f"Invalid epochs: {ep}. Must be a positive integer.")
        
        return hl, bs, ep, lr, wd, t_proportion

    # Checkpoint and run state management methods
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

    def __load_or_initialize_run(self, run: optuna.Trial, device: torch.device) -> RunState:
        try:
            state: RunState = self.__try_resume_from_checkpoint(run, device)
            self.logger.info(f"Resumed from checkpoint at epoch {state.epoch}")
        except NoCheckpointAvailable:
            # Initialize the run config from scratch (hyperparameters)
            hyperparameters = self.prepare_hyperparameters(run, device)
            hl, bs, ep, lr, wd, t_proportion = self.__check_run_config(*hyperparameters)
            
            # Build checkpoints folder path if does not exist
            checkpoint_dir = Path(self._output_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Initialize the run state from scratch
            state = RunState(
                checkpoint_path=checkpoint_dir / f"{self.study_id}_{run.number}.pkl",
                input_dim=self._input_dim,
                output_dim=self._output_dim,
                hidden_dims=hl,
                learning_rate=lr,
                weight_decay=wd,
                train_proportion=t_proportion,
                batch_size=bs,
                epochs=ep,
                epoch=0,
                best_model_loss=float("inf"),
                best_model_state_dict={},
                model_state_dict={},
                optimizer_state_dict={},
                wandb_id=f"{self.study_id}_{run.number}",
                wandb_name=f"run_{run.number}_bs{bs}_hl{'x'.join(map(str, hl))}",
            )

            # Store run information in the optuna database attributes
            for k, v in state.to_database_dict().items():
                run.set_user_attr(k, str(v) if isinstance(v, Path) else v)
        except (KeyError, RuntimeError, ValueError) as e:
            # Crash early to avoid proceeding with invalid state
            self.logger.error(f"Checkpoint is corrupted or incomplete: {e}")
            raise
        return state

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
            lambda trial: self.__run_trial(trial, *args),
            n_trials=self._num_runs,
            callbacks=[
                MaxTrialsCallback(  # This callback will stop the study once the number of trials is reached
                    self._num_runs, states=(optuna.trial.TrialState.COMPLETE,)
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
        self._num_processes = n_processes

        # Check if n_processes if higher than 0
        if self._num_processes <= 0:
            raise RuntimeError(
                "Specified number of processes must be equal or greater than 1."
            )

        # Check if n_processes is equal to 1
        if self._num_processes == 1:
            self.logger.debug("Running in single-process mode.")
            self.run(*args)
            return

        # Initialize processes
        processes: List[multiprocessing.Process] = []
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

        self.logger.info(f"Spawning {self._num_processes} parallel processes.")

        # Start processes
        for _ in range(self._num_processes):
            p = multiprocessing.Process(target=self.run, args=args)
            processes.append(p)
            p.start()
            self.logger.debug(f"Process {p.pid} started.")

        # Wait processes to complete
        for p in processes:
            p.join()
            self.logger.debug(f"Process {p.pid} finished.")

    def __run_trial(self, run: optuna.Trial, *args) -> float:
        # Initialize the run logger
        self.__init_run_logger(run)

        # Setup PyTorch device
        if torch.cuda.is_available():
            device_index = run.number % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_index}")
            self.logger.info(f"Using '{device}' CUDA device.")
        else:
            device = torch.device("cpu")
            self.logger.info(f"Using CPU.")

        # Resume or initialize the run state
        state = self.__load_or_initialize_run(run, device)

        # Call prepare_run if it exists in the subclass
        self.prepare_run(run, state, device)

        # Initialize WandB run | REFERENCE FOR DISTRIBUTED: https://docs.wandb.ai/support/multiprocessing_eg_distributed_training/
        wandb_run = wandb.init(
            id=state.wandb_id,
            project=self.study_id,
            name=state.wandb_name,
            group="neuraltrain",
            config={
                "input_dim": self._input_dim,
                "output_dim": state.output_dim,
                "hidden_dims": state.hidden_dims,
                "batch_size": state.batch_size,
                "train_portion": state.train_proportion,
                "learning_rate": state.learning_rate,
                "weight_decay": state.weight_decay,
            },
            reinit=True,
            dir=self._output_dir,
        )

        # Instantiate model and load state if available
        self.logger.debug(f"Initializing model with input_dim: {self._input_dim}, hidden_dims: {state.hidden_dims}, output_dim: {self._input_dim}.")
        model = self._model_class(
            self._input_dim, 
            state.hidden_dims, 
            self._output_dim
        ).to(device)
        if state.model_state_dict:
            model.load_state_dict(state.model_state_dict)

        # Instantiate optimizer and load state if available
        self.logger.debug(f"Initializing optimizer with learning_rate: {state.learning_rate}, weight_decay: {state.weight_decay}.")
        optimizer: torch.optim.Optimizer = self._optimizer_class(
            model.parameters(),
            lr=state.learning_rate,
            weight_decay=state.weight_decay,
        )
        if state.optimizer_state_dict:
            optimizer.load_state_dict(state.optimizer_state_dict)

        # NN training loop
        for epoch in range(state.epoch, state.epochs, 1):
            # Train the model for one epoch
            train_metrics = self.train(
                model, optimizer, device
            )
            eval_loss, eval_metrics = self.evaluate(model, device)

            # Add prefixes to metrics keys (for separation in WandB)
            train_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
            eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}

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
            Path(self._output_dir) / f"{state.wandb_name}_best_model.pth"
        )
        torch.save(state.best_model_state_dict, model_path)

        self.logger.info("=" * 70)
        self.logger.info(f"✅ Run {run.number} completed")
        self.logger.info(f" Best model saved at: {model_path}")
        self.logger.info(f" Best validation loss: {state.best_model_loss:.6f}")
        self.logger.info(f" Best model hyperparameters:")
        self.logger.info(f"   • Input Dimension: {self._input_dim}")
        self.logger.info(f"   • Output Dimension: {self._output_dim}")
        self.logger.info(f"   • Hidden Layers Sizes: {state.hidden_dims}")
        self.logger.info(f"   • Batch Size: {state.batch_size}")
        self.logger.info(f"   • Learning Rate: {state.learning_rate}")
        self.logger.info(f"   • Weight Decay: {state.weight_decay}")
        self.logger.info(f"   • Train Proportion: {state.train_proportion}")
        self.logger.info(f"   • Epochs: {state.epochs}")
        self.logger.info("=" * 70)

        wandb_run.finish()
        return state.best_model_loss

    # Abstract methods to be implemented by subclasses
    def prepare_hyperparameters(self, run: optuna.Trial, device: torch.device) -> HyperParameter:
        """
        Prepare the hyperparameters for the run based on the search space defined in the study.
        This method can be overridden in subclasses to customize hyperparameter preparation (e.g. to include Optuna suggestions).

        ### Args:
        - run (optuna.Trial): The Optuna trial object used to suggest values.
        - device (torch.device): The torch device where computations will take place.
        """
        return self.search_space[run.number]

    def prepare_run(self, run: optuna.Trial, state: RunState, device: torch.device) -> None:
        """
        Method executed right after initialization and just before the training loop starts. Can be used for set up in subclasses.

        ### Args:
        - run (optuna.Trial): The Optuna trial object used to suggest values.
        - state (RunState): The current run state containing hyperparameters and training progress.
        - device (torch.device): The torch device where computations will take place.
        """
        pass

    @abstractmethod
    def train(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
        """
        Train the model for one epoch and return training metrics.

        ### Args:
        - model (torch.nn.Module): The neural network model to be trained.
        - optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        - device (torch.device): The device on which the model is trained (CPU or GPU).

        ### Returns:
        - Dict[str, float]: A dictionary containing training metrics, such as MAE, MSE, R2, etc.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")
    
    @abstractmethod
    def evaluate(self, model: torch.nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model on the validation set and return evaluation metrics.

        ### Args:
        - model (torch.nn.Module): The neural network model to be evaluated.
        - device (torch.device): The device on which the model is evaluated (CPU or GPU).

        ### Returns:
        - Tuple[float, Dict[str, float]]: A tuple containing:
            - float: The evaluation loss (e.g., MSE). This will be used to determine the best model.
            - Dict[str, float]: A dictionary containing evaluation metrics, such as MAE, MSE, R2, etc.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")

    # Make relevant attributes accessible through property getters
    @property
    def study_id(self) -> str:
        """Returns the unique identifier for the study."""
        return self._study_id
    
    @property
    def search_space(self) -> SearchSpace:
        """Returns the search space for hyperparameters. ((hidden_dim_1, hidden_dim_2, ...), batch_size, epochs, learning_rate, weight_decay)"""
        return self._search_space.copy()
    
    @property
    def model_class(self) -> ModelType:
        """Returns the class of the model to be trained."""
        return self._model_class
    
    @property
    def optimizer_class(self) -> OptimizerType:
        """Returns the class of the optimizer to be used for training."""
        return self._optimizer_class
    
    @property
    def input_dim(self) -> int:
        """Returns the input dimension of the model."""
        return self._input_dim
    
    @property
    def output_dim(self) -> int:
        """Returns the output dimension of the model."""
        return self._output_dim
    
    @property
    def debug_mode(self) -> bool:
        """Returns whether the framework is running in debug mode."""
        return self._debug_mode
    