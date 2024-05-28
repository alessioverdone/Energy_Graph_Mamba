from time import strftime


# Hperparameters
class Configuration:
    NAME_PROJECT = 'Prova'
    NAME_RUN = 'Run_' + strftime("%d/%m/%y") + '_' + strftime("%H:%M:%S")

    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 1e-3
    LR_ITER = 30
    DECAY_LR = 0.9
    LAGS = 24
    PREDICTION_WINDOW = 24
    NUM_STATION = 0
    LEN_TRAIN = 0
    LEN_VAL = 0
    LEN_TEST = 0

    # Model Parameters
    MODEL = "GConvLSTM"  # ["GConvLSTM" , "LSTM", "CNN1D"]
    NODE_FEATURES = 24
    FILTERS = 128
    FILTER_SIZE = 2
    DROPOUT = 0.1

    # MLP Parameters
    INPUT_MLP_DIMENSION = FILTERS
    OUTPUT_MLP_DIMENSION_IMPUTATION = PREDICTION_WINDOW
    OUTPUT_MLP_DIMENSION_FORECASTING = PREDICTION_WINDOW
    INPUT_FEATURE_DIMENSION = 5

    # LSTM Parameters
    HIDDEN_DIMENSION_SINGLE = 30
    NUMBER_LSTM_LAYERS = 3

    # Use case
    FORECASTING = False
    MULTIVARIATE = False

    # Training
    USER_WANDB = "alessio_v"
    SAVE_IMGS = False
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
    LOGGER = None
    USE_LOGGER = True
    GRADIENT_CLIP = 0
    ACCUMULATE_GRADIENT_BATCHES = 1
    LIMIT_TRAIN_BATCHES = 1.0
    LIMIT_VAL_BATCHES = 1.0
    LIMIT_TEST_BATCHES = 1.0
    CHECK_VAL_EVERY_N_EPOCH = 5
    PRECISION = 32
    DETERMINISTIC = False
    ACCELERATOR = "gpu"
    ENABLE_CHECKPOINTING = False

    # Dataset
    DATA_MAP = {'PV4': '../data/Generated_time_series_output_with_weigth_light_multivariate'
                       '.json',
                'PV31': '../data/Generated_time_series_output_with_weigth_multivariate_T50'
                        '.json',
                'PV31T': '../data/Generated_time_series_output_31_with_weigth_multivariate_and_time'  # 
                         '.json',
                'PV10': '../data/Real_time_series_output_3Months_with_weigth_multivariate_T150.json'}

    # Path
    PATH_DATASET = DATA_MAP['PV4']

    save_checkpoint = False
    LOAD_PATH = "../checkpoints/epoch=51-step=83719.ckpt"
    LOAD_CHECKPOINT = False
    train_val_and_test_ratio = 0.7
    limit_time_series = False
    max_time_step = 2184
    time_series_step = 8  # less than 2 there is risk for over-fitting
