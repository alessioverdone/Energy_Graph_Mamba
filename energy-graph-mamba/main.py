import time
from time import strftime
import random
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from dataset import DataModule
from train import TrainingModule
from config import Configuration
from utils import get_hyperparams_dict, get_run_name

params = Configuration()

# Dataset
db = 'PV31T'  # ['PV31T']
params.PATH_DATASET = params.DATA_MAP[db]
params.MODEL = "CNN1D"  # ["GConvLSTM" , "LSTM", "CNN1D"]

# Run
params.EXP_NAME = get_run_name(db, params.LAGS, params.PREDICTION_WINDOW, params)

if params.DEBUG:
    params.EPOCHS = 30
    params.LIMIT_TRAIN_BATCHES = 100
    params.LIMIT_VAL_BATCHES = 100
    params.LIMIT_TEST_BATCHES = 100
    params.LOGGER = False
    params.limit_time_series = True


if params.REPRODUCIBLE:
    torch.manual_seed(params.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(params.SEED)
    np.random.seed(params.SEED)
    seed_everything(params.SEED)

start = time.time()  # get initial time

# Load datamodule
data_module = DataModule(params)
params.NUM_STATION = data_module.num_station
params.LEN_TRAIN, params.LEN_VAL, params.LEN_TEST = data_module.get_len_loader()
print('Number of PV plants: ', str(params.NUM_STATION))
params.NAME_RUN = params.EXP_NAME + '__' + strftime("%d/%m/%y") + '_' + strftime("%H:%M:%S")
print("Run name: ", params.NAME_RUN)

# Load model
model = TrainingModule(params)
params_dict = get_hyperparams_dict(params)

# Checkpoints
if params.save_checkpoint:
    checkpoint_callback = [ModelCheckpoint(dirpath="checkpoints", save_top_k=2, monitor="val_loss")]
else:
    checkpoint_callback = []

# Logger
if params.USE_LOGGER:
    wandb.login()
    wandb.init(project=params.NAME_PROJECT, name=params.NAME_RUN, entity=params.USER_WANDB, config=params_dict)
    params.LOGGER = WandbLogger(log_model=False)

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


if params.LOAD_CHECKPOINT:
    model = TrainingModule.load_from_checkpoint(params.LOAD_PATH, params=params)

# Training
trainer.fit(model, data_module)
loss_dict = trainer.test(model, data_module)

if params.USE_LOGGER:
    wandb.finish()
end = time.time()
total_time = (end - start)/60
print('Total time of training: ', str(total_time))
