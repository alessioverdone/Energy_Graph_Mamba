from utils import *
from models import *
from train import *
import time
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
from time import strftime
from pytorch_lightning import seed_everything


# System hyperparameters
class HyperParameters():
    NAME_PROJECT = 'Prova'
    NAME_RUN = 'Run_' + strftime("%d/%m/%y") + '_' + strftime("%H:%M:%S")

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 300
    LR = 1e-3
    LR_ITER = 10
    DECAY_LR = 0.9
    LAGS = 24
    PREDICTION_WINDOW = 24
    NUM_STATION = 0
    LEN_TRAIN = 0
    LEN_VAL = 0
    LEN_TEST = 0

    # Model Parameters
    GNN_MODEL = "GConvLSTM"
    NODE_FEATURES = 24  # LAGS  # 32
    FILTERS = 128  # 16
    FILTER_SIZE = 2
    DROPOUT = 0.1

    # MLP Parameters
    INPUT_MLP_DIMENSION = FILTERS
    OUTPUT_MLP_DIMENSION_IMPUTATION = PREDICTION_WINDOW
    OUTPUT_MLP_DIMENSION_FORECASTING = PREDICTION_WINDOW
    INPUT_FEATURE_DIMENSION = 5

    #LSTM Parameters
    HIDDEN_DIMENSION_SINGLE = 30
    NUMBER_LSTM_LAYERS = 3


    # Use case
    FORECASTING = False
    MULTIVARIATE = False

    # Training
    SAVE_IMGS = True
    DEBUG = False
    REPRODUCIBLE = False
    SEED = 42
    NUM_GPUS = 1
    if NUM_GPUS == 1:
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    NUM_WORKERS = 2  # multiprocessing.cpu_count()
    FAST_DEV_RUN = False
    LOGGER = True
    BAR_REFRESH_RATE = 1
    GRADIENT_CLIP = 0
    ACCUMULATE_GRADIENT_BATCHES = 1
    LIMIT_TRAIN_BATCHES = 1.0  # 1.0 # 1500
    LIMIT_VAL_BATCHES = 1.0  # 1.0 # 400
    LIMIT_TEST_BATCHES = 1.0
    AUTO_LR_FIND = False
    CHECK_VAL_EVERY_N_EPOCH = 5

    # Dataset
    DATA_MAP1 = {'PV4': 'H:\Il mio Drive\PhD ICT\Data\Generated_time_series_output_with_weigth_light_multivariate'
                       '.json',
                'PV31': 'H:\Il mio Drive\PhD ICT\Data\Generated_time_series_output_with_weigth_multivariate_T50'
                        '.json',
                'PV31T': 'H:\Il mio Drive\PhD ICT\Data\Generated_time_series_output_31_with_weigth_multivariate_and_time'# 
                         '.json',
                'PV10': 'H:\Il mio Drive\PhD ICT\Data\Real_time_series_output_3Months_with_weigth_multivariate_T150.json'}

    # Dataset
    DATA_MAP2 = {'PV4': 'G:\Il mio Drive\PhD ICT\Data\Generated_time_series_output_with_weigth_light_multivariate'
                       '.json',
                'PV31': 'G:\Il mio Drive\PhD ICT\Data\Generated_time_series_output_with_weigth_multivariate_T50'
                        '.json',
                'PV31T': 'G:\Il mio Drive\PhD ICT\Data\Generated_time_series_output_31_with_weigth_multivariate_and_time'    
                         '.json',
                'PV10': 'G:\Il mio Drive\PhD ICT\Data\Real_time_series_output_3Months_with_weigth_multivariate_T150.json'}

    # Path
    PATH_DATASET = DATA_MAP2['PV4']

    # Other
    NOTES = ""


params = HyperParameters()
results = []

###################  EXPERIMENT SETUP #################################################################################
params.DEBUG = False
params.NAME_PROJECT = 'TEST_NOTTE'
params.EPOCHS = 200
params.GNN_MODEL = "GNN"

# Window
in_ws = 24
params.LAGS = in_ws
out_win = 24
params.PREDICTION_WINDOW = out_win
params.OUTPUT_MLP_DIMENSION_FORECASTING = out_win

# Task
params.FORECASTING = True
params.MULTIVARIATE = True
if params.MULTIVARIATE:
    params.INPUT_FEATURE_DIMENSION = 5

# Dataset
db = 'PV31T'
params.PATH_DATASET = params.DATA_MAP1[db]
#params.PATH_DATASET = 'H:\Il mio Drive\PhD ICT\Data\Generated_time_series_output_29_with_weigth_multivariate_and_time.json'


# Runs
RUNS = 1
params.REPRODUCIBLE = True
params.EXP_NAME = get_run_name(db, in_ws, out_win, params)
params.LOGGER = False
#######################################################################################################################

IN_WIN = [24]
OUT_WIN = [24]

if params.DEBUG:
    params.EPOCHS = 30
    params.LIMIT_TRAIN_BATCHES = 100
    params.LIMIT_VAL_BATCHES = 100
    params.LIMIT_TEST_BATCHES = 100
    params.LOGGER = False

for i in range(len(IN_WIN)):
    for j in range(len(OUT_WIN)):

        if params.REPRODUCIBLE:
            torch.manual_seed(params.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            random.seed(params.SEED)
            np.random.seed(params.SEED)
        else:
            seed = random.randint(1, 10000)
            params.SEED = seed
            torch.manual_seed(params.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(params.SEED)
            random.seed(params.SEED)
            np.random.seed(params.SEED)
            seed_everything(params.SEED)
            torch.use_deterministic_algorithms(True)

        params.LAGS = IN_WIN[i]
        params.PREDICTION_WINDOW = OUT_WIN[j]

        start = time.time()
        data_module = DataModule(params)
        params.NUM_STATION = data_module.num_station
        params.LEN_TRAIN, params.LEN_VAL, params.LEN_TEST = data_module.get_len_loader()
        print('Number of station: ', str(params.NUM_STATION))
        params.NAME_RUN = params.EXP_NAME + '__' + strftime("%d/%m/%y") + '_' + strftime("%H:%M:%S")
        print("Run name: ", params.NAME_RUN)
        model = TrainingModule(params)
        params_dict = get_hyperparams_dict(params)
        #checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=2, monitor="val_loss_FM_power")

        # Logger
        if params.LOGGER:
            wandb.login()
            wandb.init(project=params.NAME_PROJECT, name=params.NAME_RUN, entity="alessio_v", config=params_dict)
            params.LOGGER = WandbLogger(log_model=False)
            # wandb_logger.watch(model, log_freq=100)  # log='gradients',

        trainer = pl.Trainer(
            max_epochs=params.EPOCHS,
            fast_dev_run=params.FAST_DEV_RUN,
            logger=params.LOGGER,
            progress_bar_refresh_rate=params.BAR_REFRESH_RATE,
            gpus=params.NUM_GPUS,
            gradient_clip_val=params.GRADIENT_CLIP,
            check_val_every_n_epoch=params.CHECK_VAL_EVERY_N_EPOCH,
            accumulate_grad_batches=params.ACCUMULATE_GRADIENT_BATCHES,
            limit_train_batches=params.LIMIT_TRAIN_BATCHES,
            limit_val_batches=params.LIMIT_VAL_BATCHES,
            limit_test_batches=params.LIMIT_TEST_BATCHES,
            auto_lr_find=params.AUTO_LR_FIND)#, callbacks=[checkpoint_callback]
            # precision=32
            # deterministic=True
        #)

        # LOAD_PATH = r'H:\Il mio Drive\PhD ICT\Code\Fase_2\Projects\test_architettura_2\test_6\checkpoints\epoch=51-step=83719.ckpt'  #BS=64, FS=2, EP=4
        # model = TrainingModule.load_from_checkpoint(LOAD_PATH, params=params)
        # trainer.test(model, data_module)
        # trainer.tune(model, data_module)
        trainer.fit(model, data_module)
        loss_dict = trainer.test(model, data_module)
        results.append(loss_dict[0])
        if params.LOGGER: wandb.finish()
        end = time.time()
        total_time = (end - start)/60
        print('Total time of training: ', str(total_time))
        val_loss_trend = model.val_loss_during_training
        print(model.val_loss_during_training)
        axis_time = plot_and_save_loss(val_loss_trend, total_time, params)
        plt.show()
        final_tensor = torch.zeros(2,len(val_loss_trend))
        final_tensor[0,:] = torch.tensor(val_loss_trend)
        final_tensor[1,:] = torch.tensor(axis_time)
        torch.save(final_tensor, 'loss_time_tensor_LSTM.pt')

    print_runs_results(params, RUNS, results)
