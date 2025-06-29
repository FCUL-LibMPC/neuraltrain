import torch
import pandas as pd
from typing import Callable, List
from torch.utils.data import Dataset, DataLoader, Subset


class BasicDataset(Dataset):
    def __init__(self, 
        df: pd.DataFrame, 
        input_cols: list, 
        target_cols: list,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initializes the BasicDataset with input and target columns.

        Args:
            df (pandas.DataFrame): The dataframe containing the data.
            input_cols (list): Ordered list of column names to be used as input features.
            target_cols (list): Ordered list of column names to be used as target variables.
        """
        # Save the input and target columns
        self.__input_cols = input_cols.copy()
        self.__target_cols = target_cols.copy()

        # Convert input and target data into PyTorch tensors
        self.input_data = torch.tensor(df[self.__input_cols].values, dtype=torch.float32, device=device)
        self.target_data = torch.tensor(df[self.__target_cols].values, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        if self.target_data is not None:
            return self.input_data[idx], self.target_data[idx]
        return self.input_data[idx]

    def __repr__(self):
        return (
            f"<TimeSeriesDataset>\n"
            f"  Samples     : {len(self)}\n"
            f"  Input Dim   : {len(self.__input_cols)}\n"
            f"  Target Dim  : {len(self.__target_cols)}\n"
            f"  Input Cols  : {self.__input_cols}\n"
            f"  Target Cols : {self.__target_cols}\n"
        )

    @property
    def input_cols(self):
        return self.__input_cols.copy()

    @property
    def target_cols(self):
        return self.__target_cols.copy()

    @classmethod
    def get_dataloaders(
        cls,
        df: pd.DataFrame,
        input_cols: List[str],
        target_cols: List[str],
        train_proportion: float = 0.8,
        batch_size: int = 254,
        shuffle: bool = True,
        random_state: int = 2,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Splits the dataframe into training and testing sets, creates PyTorch datasets, and returns dataloaders.

        #### Args:
        - df (pandas.DataFrame): The input dataframe containing the data.
        - input_col_filter (Callable[[str], bool]): A function to filter input columns based on their names.
        - target_col_filter (Callable[[str], bool]): A function to filter target columns based on their names.
        - train_proportion (float, optional): The proportion of the data to be used for training. Defaults to 0.8.
        - batch_size (int, optional): The number of samples per batch to load. Defaults to 254.
        - shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
        - random_state (int, optional): The seed used by the random number generator for shuffling. Defaults to 2.

        Returns:
            tuple[DataLoader, DataLoader]: A tuple containing the training and testing dataloaders.
        """
        if shuffle:
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_df = df[: int(train_proportion * len(df))]
        test_df = df[int(train_proportion * len(df)) :]

        train_ds = cls(train_df, input_cols, target_cols)
        test_ds = cls(test_df, input_cols, target_cols)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_dl, test_dl

class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        period: int,
        input_cols: List[str],
        target_cols: List[str],
        input_window_size: int
    ):
        """
        Initializes the TimeSeriesDataset with input and target columns.

        #### Args:
        - df (pandas.DataFrame): The dataframe containing the time series data.
        - period (int): The time period (in seconds) for each step in the time series.
        - input_cols (List[str]): Ordered list of column names to be used as input features.
        - target_cols (List[str]): Ordered list of column names to be used as target variables.
        - input_window_size (int): The size in hours for the input array required for a prediction. This will be used to guarantee that the model has enough data to make a prediction for the first step.
        """
        self.steps_per_hour = int(3600 / period)
        self.input_cols = input_cols.copy()
        self.target_cols = target_cols.copy()
        self.input_window_size = int(self.steps_per_hour * input_window_size) + 1

        relevant_columns = list(dict.fromkeys(input_cols + target_cols))
        self.df = df[relevant_columns].reset_index(drop=True)

        self.num_samples = len(self.df) - self.input_window_size
        self.num_input_features = len(self.input_cols)
        self.num_target_features = len(self.target_cols)

        # Alocar tensores
        self.input_tensor = torch.zeros((self.num_samples, self.input_window_size, self.num_input_features), dtype=torch.float32)
        self.target_tensor = torch.zeros((self.num_samples, self.num_target_features), dtype=torch.float32)

        # Preencher tensores
        for i in range(self.num_samples):
            input_window = self.df[self.input_cols].iloc[i : i + self.input_window_size].values
            target_values = self.df[self.target_cols].iloc[i + self.input_window_size].values

            self.input_tensor[i] = torch.tensor(input_window, dtype=torch.float32)
            self.target_tensor[i] = torch.tensor(target_values, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.input_tensor[idx], self.target_tensor[idx]

    @classmethod
    def get_dataloaders(
        cls,
        df: pd.DataFrame,
        period: int,
        input_cols: List[str],
        target_cols: List[str],
        input_window_size: int,
        train_proportion: float = 0.8,
        batch_size: int = 254,
        shuffle: bool = True,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[DataLoader, DataLoader]:
        """
        Builds a full dataset and splits it using torch.utils.data.Subset.

        ### Args:
        - df (pd.DataFrame): The full input dataframe.
        - period (int): Time period in seconds for each step.
        - input_cols (List[str]): Names of input features.
        - target_cols (List[str]): Names of target features.
        - input_window_size (int): Window size in hours for inputs.
        - train_proportion (float): Proportion of data to use for training.
        - batch_size (int): Batch size.
        - shuffle (bool): Whether to shuffle before split.
        - seed (int): Random seed for reproducibility.
        - device (torch.device): Device to store tensors (e.g., 'cuda').

        Returns:
            tuple[DataLoader, DataLoader]: train and test dataloaders
        """
        # Create full dataset (eager tensors already built)
        full_dataset = cls(
            df=df,
            period=period,
            input_cols=input_cols,
            target_cols=target_cols,
            input_window_size=input_window_size,
            device=device,
        )

        num_samples = len(full_dataset)
        indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(num_samples)

        split_idx = int(train_proportion * num_samples)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_subset = Subset(full_dataset, train_indices)
        test_subset = Subset(full_dataset, test_indices)

        train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
        test_dl = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        return train_dl, test_dl, train_subset, test_subset

class RobustnessEvalDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        period: int,
        input_cols: List[str],
        target_cols: List[str],
        step_size: int,
        input_window_size: int,
        prediction_window_size: int,
        analysis_horizon: int
    ):
        """
        Initializes the RecursiveEvalDataset with input and target columns.
        This dataset is designed for recursive long-horizon evaluation, where the model predicts future values 
        by feeding back its own predictions to achieve longer horizon predictions.

        ### Args:
        - df (pandas.DataFrame): The dataframe containing the time series data.
        - period (int): The time period (in seconds) for each step in the time series.
        - input_cols (List[str]): Ordered list of column names to be used as input features.
        - target_cols (List[str]): Ordered list of column names to be used as target variables.
        - step_size (int): The number of hours between the start of each regression window.
        - input_window_size (int): The size in hours for the input array required for a prediction. This will be used to guarantee that the model has enough data to make a prediction for the first step.
        - prediction_window_size (int): The size in hours for the next prediction window. This will be used to guarantee that the model has enough data to be used in the rest of the evaluation.
        - analysis_horizon (int): The total number of days of the analysis period. This will be used to determine the data window that will be used from the dataset.
        """
        self.steps_per_hour = int(3600 / period)
        self.input_cols = input_cols.copy()
        self.target_cols = target_cols.copy()
        self.step_size = int(self.steps_per_hour * step_size)
        self.input_window_size = int(self.steps_per_hour * input_window_size) + 1
        self.prediction_window_size = int(self.steps_per_hour * prediction_window_size)
        self.window_size = self.input_window_size + self.prediction_window_size

        relevant_columns = list(dict.fromkeys(input_cols + target_cols))
        self.df = df[relevant_columns].reset_index(drop=True).iloc[: int(analysis_horizon * 24 * self.steps_per_hour)]

        self.num_samples = (len(self.df) - self.window_size) // self.step_size
        self.num_input_features = len(self.input_cols)
        self.num_target_features = len(self.target_cols)

        # Alocar tensores
        self.input_tensor = torch.zeros((self.num_samples, self.window_size, self.num_input_features), dtype=torch.float32)
        self.target_tensor = torch.zeros((self.num_samples, self.prediction_window_size, self.num_target_features), dtype=torch.float32)

        # Preencher tensores
        for i in range(self.num_samples):
            idx = i * self.step_size
            input_window = self.df[self.input_cols].iloc[idx : idx + self.window_size].values
            target_values = self.df[self.target_cols].iloc[idx + self.input_window_size + 1 : idx + self.window_size + 1].values

            self.input_tensor[i] = torch.tensor(input_window, dtype=torch.float32)
            self.target_tensor[i] = torch.tensor(target_values, dtype=torch.float32)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.input_tensor[idx].clone(), self.target_tensor[idx].clone(), int(self.input_window_size)
    
    @classmethod
    def get_dataloader(
        cls,
        df: pd.DataFrame,
        period: int,
        input_cols: List[str],
        target_cols: List[str],
        step_size: int = 1,
        input_window_size: int = 24,
        prediction_window_size: int = 24,
        analysis_horizon: int = 183,
        shuffle: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> DataLoader:
        """
        Returns a DataLoader for recursive long-horizon evaluation over the full dataset.

        ### Args:
        - df (pd.DataFrame): The dataframe containing the time series data.
        - period (int): The time period (in seconds) for each step in the time series.
        - input_cols (List[str]): Ordered list of column names to be used as input.
        - target_cols (List[str]): Ordered list of column names to be used as target variables.
        - step_size (int): The number of hours between the start of each regression window.
        - input_window_size (int): The size in hours for the amount of historical data required to make a prediction.
        - prediction_window_size (int): The size in hours for the next prediction window.
        - analysis_horizon (int): The total number of days of the analysis period.
        - shuffle (bool): Tells torch to shuffle the dataset when iterating over it.
        - device (torch.device): The device to store tensors (e.g., 'cuda' or 'cpu').

        ### Returns:
        - DataLoader: A PyTorch DataLoader yielding (input, target) tuples.
        """

        dataset = cls(
            df=df,
            period=period,
            input_cols=input_cols,
            target_cols=target_cols,
            step_size=step_size,
            input_window_size=input_window_size,
            prediction_window_size=prediction_window_size,
            analysis_horizon=analysis_horizon,
            device=device,
        )

        batch_size = len(dataset)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset
    
if __name__ == "__main__":
    df = pd.read_csv("library2024_simulation_dataset_normalized.csv", index_col="Timestamp", parse_dates=True)
    input_cols = ["Tz", "Tout", "WindSpd", "GHI", "Thvac", "RTUsouth", "win1", "win2", "win3", "win5"]
    target_cols = ["Tz", "Ehvac"]

    dataset = RobustnessEvalDataset(
        df=df,
        period=300,
        input_cols=input_cols,
        target_cols=target_cols,
        step_size=1,
        input_window_size=24,
        prediction_window_size=24,
        analysis_horizon=183,
    )

    inputs, targets = dataset[0]
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
    print(inputs)
    print(targets)

if __name__ == "__main__3132131":
    df = pd.read_csv("library2024_simulation_dataset_normalized.csv", index_col="Timestamp", parse_dates=True)
    input_cols = ["Tz", "Tout", "WindSpd", "GHI", "Thvac", "RTUsouth", "win1", "win2", "win3", "win5"]
    target_cols = ["Tz", "Ehvac"]

    dataset = TimeSeriesDataset(
        df=df,
        period=300,
        input_cols=input_cols,
        target_cols=target_cols,
        input_window_size=24,
    )

if __name__ == "__main__dasdsa":
    from torchmetrics.regression import MeanAbsoluteError

    # TEST RecursiveEvalDataset WITH DUMMY DATA

    df = pd.read_csv("library2024_simulation_dataset_normalized.csv", index_col="Timestamp")

    input_cols = ["Tz", "Tout", "WindSpd", "GHI", "Thvac", "RTUsouth", "RTUnorth"]
    target_cols = ["Tz", "Ehvac"]

    recursive_test_dl = RobustnessEvalDataset.get_dataloader(
        df=df,
        period=300,
        input_cols=input_cols,
        target_cols=target_cols,
    )
    recursive_test_dataset: RobustnessEvalDataset = recursive_test_dl.dataset

    inputs, targets, input_start_idx = next(iter(recursive_test_dl))
    inputs = inputs.squeeze(0)   # shape: [timestep, input_cols]
    targets = targets.squeeze(0) # shape: [timestep, target_cols]

    tz_index = input_cols.index("Tz")           #recursive_test_dataset.get_input_col_index("Tz")
    tz_target_index = target_cols.index("Tz")   #recursive_test_dataset.get_target_col_index("Tz")

    print("inputs:", inputs)
    print("targets:", targets)
    print("input_start_idx:", input_start_idx)

    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}, t: {input_start_idx.item()}, Horizon: {len(targets)}")
    print(f"Loop range: {0} to {len(targets) - 1}")
    print(f"Input range: {input_start_idx.item()} to {input_start_idx.item() + len(targets)}")

    # Simulate dummy predictions: just add +1 to previous Tz
    predictions = []
    for i in range(len(targets)):
        # Use last known Tz or previous prediction
        if i == 0:
            prev_tz = inputs[input_start_idx + i - 1, tz_index]
        else:
            prev_tz = predictions[i - 1]

        prediction = targets[i, tz_target_index] + 1.0
        predictions.append(prediction)

        # Feedback into input for future prediction
        inputs[input_start_idx + i, tz_index] = prediction

    # Convert to tensor
    predictions_tensor = torch.stack(predictions)

    # Evaluate error
    mae = MeanAbsoluteError()
    error = mae(predictions_tensor.unsqueeze(1), targets[:, tz_target_index].unsqueeze(1))
    print(f"Simulated MAE on dummy recursive test: {error.item():.4f}")

    # TEST __len__ AND __getitem__ FOR CORRECTNESS
    del recursive_test_dl

    recursive_test_dl = RobustnessEvalDataset.get_dataloader(
        df=df,
        period=300,
        input_cols=input_cols,
        target_cols=target_cols,
    )
    recursive_test_dataset: RobustnessEvalDataset = recursive_test_dl.dataset

    for i in range(len(recursive_test_dataset)):
        try:
            input_tensor, target_tensor, input_start_t = recursive_test_dataset[i]
            assert input_tensor.shape[0] >= input_start_t, f"Invalid input slice shape at i={i}"
        except Exception as e:
            print(f"Error at index {i}: {e}")
            break
    else:
        print("All indices are safe.")