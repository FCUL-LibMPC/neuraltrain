import torch
import pandas as pd
from typing import Callable
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_cols: list, target_cols: list):
        """
        Initializes the TimeSeriesDataset with input and target columns.

        Args:
            df (pandas.DataFrame): The dataframe containing the time series data.
            input_cols (list): Ordered list of column names to be used as input features.
            target_cols (list): Ordered list of column names to be used as target variables.
        """
        # Save the input and target columns
        self.__input_cols = input_cols.copy()
        self.__target_cols = target_cols.copy()

        # Convert input and target data into PyTorch tensors
        self.input_data = torch.tensor(
            df[self.__input_cols].values, dtype=torch.float32
        )
        self.target_data = torch.tensor(
            df[self.__target_cols].values, dtype=torch.float32
        )

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
        input_col_filter: Callable[[str], bool],
        target_col_filter: Callable[[str], bool],
        train_proportion: float = 0.8,
        batch_size: int = 254,
        shuffle: bool = True,
        random_state: int = 2,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Splits the dataframe into training and testing sets, creates PyTorch datasets, and returns dataloaders.

        Args:
            df (pandas.DataFrame): The input dataframe containing the data.
            input_col_filter (Callable[[str], bool]): A function to filter input columns based on their names.
            target_col_filter (Callable[[str], bool]): A function to filter target columns based on their names.
            train_proportion (float, optional): The proportion of the data to be used for training. Defaults to 0.8.
            batch_size (int, optional): The number of samples per batch to load. Defaults to 254.
            shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
            random_state (int, optional): The seed used by the random number generator for shuffling. Defaults to 2.

        Returns:
            tuple[DataLoader, DataLoader]: A tuple containing the training and testing dataloaders.
        """
        if shuffle:
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_df = df[: int(train_proportion * len(df))]
        test_df = df[int(train_proportion * len(df)) :]

        input_cols = [col for col in df.columns if input_col_filter(col)]
        target_cols = [col for col in df.columns if target_col_filter(col)]

        train_ds = cls(train_df, input_cols, target_cols)
        test_ds = cls(test_df, input_cols, target_cols)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_dl, test_dl
