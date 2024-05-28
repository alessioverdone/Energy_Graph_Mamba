# energy-graph-mamba


[![PyPI version](https://badge.fury.io/py/x-transformers.svg)](https://badge.fury.io/py/x-transformers)

An implementation of Mamba architecture on forecasting temporal graph time series on energy tasks.

## Install

```bash
$ conda env create -f environment.yml
```

ATTENTION: torch-geometric-temporal has a bug when installed related to "to_dense_adj" function. It can be simply resolved by commenting the line of error (torch-geometric-temporal/nn/attention/tsagcn.py (r:6)) and adding:
```python
# from torch_geometric.utils.to_dense_adj import to_dense_adj
from torch_geometric.utils import to_dense_adj
```
## Usage

For running the training you need only to run the ```main.py``` file. 
The class ```Configuration``` define the all the parameters of the model.
Models can be changed in ```main.py``` file: ```params.MODEL = "GConvLSTM"```; then all you need is to link the name of the model to its class on the ```get_model()``` function. 

The actual dataset ```PV31T``` contains time series data of photovoltaic power, wind, temperature, months and hour information of 31 simulated photovoltaic plants. The training procedure is implemented in ```pytorch-lightning```. Here there is an example of the basic implementation of the ```main.py``` function.

```python
import pytorch_lightning as pl

# Get parameters
params = Configuration()

# Dataset
db = 'PV31T'  # ['PV31T']
params.PATH_DATASET = params.DATA_MAP[db]
params.MODEL = "CNN1D"  # ["GConvLSTM" , "LSTM", "CNN1D"]

# Load datamodule
data_module = DataModule(params)

# Load model
model = TrainingModule(params)
params_dict = get_hyperparams_dict(params)

# Trainer
trainer = pl.Trainer(
    max_epochs=params.EPOCHS,
    fast_dev_run=params.FAST_DEV_RUN,
    logger=params.LOGGER,
    accelerator=params.ACCELERATOR,
    gradient_clip_val=params.GRADIENT_CLIP,
    check_val_every_n_epoch=params.CHECK_VAL_EVERY_N_EPOCH,
    accumulate_grad_batches=params.ACCUMULATE_GRADIENT_BATCHES,
    enable_checkpointing=params.ENABLE_CHECKPOINTING,
    limit_train_batches=params.LIMIT_TRAIN_BATCHES,
    limit_val_batches=params.LIMIT_VAL_BATCHES,
    limit_test_batches=params.LIMIT_TEST_BATCHES,
    callbacks=checkpoint_callback,
    precision=params.PRECISION,
    deterministic=params.DETERMINISTIC)
    
# Training
trainer.fit(model, data_module)
        return F.log_softmax(x, dim=1)
```
Images and metrics visualization can be done with Wandb logger.




