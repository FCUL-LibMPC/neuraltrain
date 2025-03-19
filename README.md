# NeuralTrain - Neural Network Training Framework

NeuralTrain is a lightweight and extensible framework for training neural networks, designed to streamline the process of hyperparameter optimization, distributed training, and checkpointing. It integrates seamlessly with **Optuna** for hyperparameter tuning and supports **multiprocessing** for  parallel training.

## Index
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

---

## Features
- **Flexible Neural Network Training** with an extendable base class (`NeuralTrainBase`).
- **Hyperparameter Optimization** using **Optuna**.
- **Parallel Training Support** with multiprocessing.
- **Automatic Trial Recovery** – failed trials (due to crashes or power outages) are re-enqueued.
- **Checkpointing** – offers tools for creating/loading checkpoints.

---

## Installation

You can install **NeuralTrain** directly from GitHub:

```sh
pip install git+https://github.com/FCUL-LibMPC/neuraltrain.git
```

---

## Usage

### **1️⃣ Extending the Base Class**
To use **NeuralTrain**, create a class that inherits from `NeuralTrainBase` and implement the `objective` method:

```python
from neuraltrain.base import NeuralTrainBase
import optuna
import torch

class MyModelTrainer(NeuralTrainBase):
    def objective(self, trial: optuna.Trial, *args) -> float:
        # Define hyperparameter search space
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
        model = torch.nn.Linear(10, hidden_dim)  # Example model

        # Implement training logic
        loss = some_training_function(model)

        return loss  # Return validation loss for Optuna optimization
```

---

### **2️⃣ Running a Training Study**
Run training with a specified number of trials:

```python
trainer = MyModelTrainer(
    db_url="sqlite:///optuna_study.db",
    study_name="my_experiment",
    n_trials=50,
    output_dir="results/"
)

trainer.run()
```

For **parallel execution**, use:

```python
trainer.distributed_run(n_processes=4)
```

---

## Contributing
Contributions, issues, and feature requests are welcome!  
Feel free to submit a PR or open an issue on GitHub.

---

## Acknowledgments
- [Optuna](https://optuna.org/) - Hyperparameter tuning.

---

**📬 Stay Updated**  
Follow **[@alxgeraldo](https://github.com/alxgeraldo)** on GitHub.

---

