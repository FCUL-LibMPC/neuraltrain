# Include src folder to Python system path
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from neuraltrain import NeuralTrainerBase, FeedForwardNN, RunConfig, parse_cli_args, torch, denormalize_temp, TimeSeriesDataset
import tyro
import pandas as pd

class OptunaHiddenLayers(NeuralTrainerBase):

    def __init__(self, input_dim, output_dim, db_url, study_id, n_runs, output_dir, model, optimizer, sampler=None, pruner=None, debug_mode=False):
        super().__init__(db_url, study_id, n_runs, output_dir, model, optimizer, sampler, pruner, debug_mode)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def setup_run(self, run) -> RunConfig:

        hidden_dims = [8, 16, 32]
        
        return RunConfig(
            hidden_dims=[hidden_dims[run.number], hidden_dims[run.number]],
            batch_size=512,
            epochs=200,
            train_proportion=0.8,
            learning_rate=0.001,
            weight_decay=0.001/200,
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )

    def loss_function(self):
        return torch.nn.MSELoss()

    def input_col_filter(self, col: str) -> bool:
        return "Tout" in col and not col.endswith("_1")

    def target_col_filter(self, col: str) -> bool:
        return col == "Tout_1"

    def denormalize(self, data):
        return denormalize_temp(data)

if __name__ == "__main__":
    args = parse_cli_args()

    # Load CSV dataset and Pytorch dataloaders
    df = pd.read_csv(args.input_file)
    model_var = "Tout"
    train_dl, test_dl = TimeSeriesDataset.get_dataloaders(
        df,
        input_col_filter=lambda col: model_var in col and col != f"{model_var}_1",
        target_col_filter=lambda col: col == f"{model_var}_1",
        train_proportion=0.8,
        batch_size=512,
    )

    # Get input and target layers dimensions
    input_dim = len(train_dl.dataset.input_cols)
    output_dim = len(train_dl.dataset.target_cols)

    # Avoid reusing dataset in the jobs processes
    del train_dl, test_dl, df

    # Start the train
    study = OptunaHiddenLayers(
        input_dim=input_dim,
        output_dim=output_dim,
        db_url="sqlite:///optuna_study.db",
        study_id=args.train_id,
        n_runs=3,
        output_dir=args.output_dir,
        model=FeedForwardNN,
        optimizer=torch.optim.Adam,
        debug_mode=args.debug_mode,
    )
    study.distributed_run(
        args.processes,
        args.input_file
    )

