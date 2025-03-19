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


class NeuralTrainerBase(ABC):
    """Abstract class to implement a Neural Network training levaraging Optuna for optimization and paralelism."""

    def __init__(
        self,
        db_url: str,
        study_name: str,
        n_trials: int,
        output_dir: Path,
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
    def objective(self, trial: optuna.Trial, *args) -> float:
        """Objective function to optimize. To be implemented in child classes."""
        ...
