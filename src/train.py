import torch
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.metrics import mean_absolute_error as MAE

from utils import *
from models import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset


class Dataset_custom(Dataset):

    def __init__(self, params):
        self.params = params
        self._read_json_data()
        self.lags = None
        self.features = None
        self.features_corrupted = None
        self.targets = None
        self.features_temperatures = None
        self.targets_temperatures = None
        self.features_winds = None
        self.targets_winds = None
        self.number_of_station = None
        self.encoded_data = []
        self.read_dataset(self.params.LAGS)

    def _read_json_data(self):
        file = self.params.PATH_DATASET
        with open(file) as f:
            self._dataset = json.load(f)

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        # Power
        stacked_target = np.stack(self._dataset["block"])
        scaler = MinMaxScaler()
        scaler.fit(stacked_target)
        standardized_target = scaler.transform(stacked_target)
        # Temperature
        stacked_temp = np.stack(self._dataset["block_temp"])
        scaler = MinMaxScaler()
        scaler.fit(stacked_temp)
        standardized_temp = scaler.transform(stacked_temp)
        # Wind
        stacked_wind = np.stack(self._dataset["block_wind"])
        scaler = MinMaxScaler()
        scaler.fit(stacked_wind)
        standardized_wind = scaler.transform(stacked_wind)
        # Month
        stacked_month = np.stack(self._dataset["block_month"])
        scaler = MinMaxScaler()
        scaler.fit(stacked_month)
        standardized_month = scaler.transform(stacked_month)
        # Hour
        stacked_hour = np.stack(self._dataset["block_hour"])
        scaler = MinMaxScaler()
        scaler.fit(stacked_hour)
        standardized_hour = scaler.transform(stacked_hour)

        self.number_of_station = stacked_target.shape[1]

        self.features = [
            np.concatenate((standardized_target[i: i + self.lags, :].T,
                            standardized_temp[i: i + self.lags, :].T,
                            standardized_wind[i: i + self.lags, :].T,
                            standardized_month[i: i + self.lags, :].T,
                            standardized_hour[i: i + self.lags, :].T), axis=-1)
            # np.concatenate((standardized_target[i: i + self.lags, :].T,
            #                 standardized_temp[i: i + self.lags, :].T,
            #                 standardized_wind[i: i + self.lags, :].T), axis=-1)

            # list of (4, 3, 24)
            for i in range(standardized_target.shape[0] - self.lags - self.params.PREDICTION_WINDOW)
        ]
        self.features = self.features[:2184]

        self.targets = [
            np.concatenate((standardized_target[i:i + self.params.PREDICTION_WINDOW, :].T,
                            standardized_temp[i:i + self.params.PREDICTION_WINDOW, :].T,
                            standardized_wind[i:i + self.params.PREDICTION_WINDOW, :].T,
                            standardized_month[i:i + self.params.PREDICTION_WINDOW, :].T,
                            standardized_hour[i:i + self.params.PREDICTION_WINDOW, :].T), axis=-1)
            # np.concatenate((standardized_target[i:i + self.params.PREDICTION_WINDOW, :].T,
            #                 standardized_temp[i:i + self.params.PREDICTION_WINDOW, :].T,
            #                 standardized_wind[i:i + self.params.PREDICTION_WINDOW, :].T), axis=-1)

            for i in range(self.lags, standardized_target.shape[0] - self.params.PREDICTION_WINDOW)
        ]
        self.targets = self.targets[:2184]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def read_dataset(self, lags) -> StaticGraphTemporalSignal:
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        for i in range(len(self.features)):
            self.encoded_data.append(Data(x=torch.FloatTensor(self.features[i]),
                                          edge_index=torch.LongTensor(self._edges),
                                          edge_attr=torch.FloatTensor(self._edge_weights),
                                          y=torch.FloatTensor(self.targets[i])))


class DataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.num_station = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        dataset = Dataset_custom(self.params)
        self.num_station = dataset.number_of_station  # len(loader.features[0])

        len_dataset = len(dataset)
        train_ratio = 0.7
        val_test_ratio = 0.5
        train_snapshots = int(train_ratio * len_dataset)
        val_test_snapshots = len_dataset - train_snapshots
        val_snapshots = int(val_test_ratio * val_test_snapshots)
        test_snapshots = len_dataset - train_snapshots - val_snapshots
        tr_db, val_db, te_db = torch.utils.data.random_split(dataset, [train_snapshots, val_snapshots, test_snapshots])
        self.train_loader = DataLoader(tr_db, batch_size=self.params.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_db, batch_size=self.params.BATCH_SIZE)
        self.test_loader = DataLoader(te_db, batch_size=self.params.BATCH_SIZE)

    # def setup(self, stage=None):

    def get_len_loader(self):
        len_train = 0
        len_val = 0
        len_test = 0
        for _ in self.train_loader: len_train += 1
        for _ in self.train_loader: len_val += 1
        for _ in self.train_loader: len_test += 1
        return len_train, len_val, len_test

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class TrainingModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # Index
        self.index = 0
        self.train_min_length = None
        self.val_min_length = None
        # Lr
        self.learning_rate = self.params.LR
        self.automatic_optimization = False

        self.val_loss_during_training = []
        self.current_val_loss = 10000
        self.val_batch_list = []
        if self.params.FORECASTING:
            # Models
            #self.GNN_Forecasting = GNN_Forecasting(params)
            self.LSTM_Forecasting = LSTM_Forecasting(params)
            #self.CNN_Forecasting = CNN_Forecasting(params)
            self.mse_f = nn.MSELoss()

    """ Index """

    def on_train_epoch_start(self):  # on_epoch_start
        self.train_min_length = min(self.params.LIMIT_TRAIN_BATCHES, self.params.LEN_TRAIN)
        self.val_min_length = min(self.params.LIMIT_VAL_BATCHES, self.params.LEN_VAL)
        self.index = 1 #np.random.randint(0, self.val_min_length - 1)

        # Updating LR
        if self.current_epoch % self.params.LR_ITER == 0 and self.current_epoch > 0:
            print('Updating learning rate at epoch: ', self.current_epoch)
            main_opt = self.optimizers()
            self.learning_rate = self.learning_rate * self.params.DECAY_LR
            for group in main_opt.param_groups:
                group["lr"] = self.learning_rate
        self.log('lr', self.learning_rate, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def on_validation_epoch_end(self):
        l = self.val_batch_list.copy()
        mean_val_loss = torch.Tensor(l).mean()
        self.val_loss_during_training.append(mean_val_loss)
        self.val_batch_list = []


    def forward(self, input, edge_index, edge_weight):
        #output = self.GNN_Forecasting(input, edge_index, edge_weight)
        output = self.LSTM_Forecasting(input)
        #output = self.CNN_Forecasting(input)
        return output

    """ Step """

    def training_step(self, train_batch, batch_idx):
        # Get data from batches
        if self.params.MULTIVARIATE:
            x = train_batch.x
        else:
            x = train_batch.x[:, :self.params.LAGS]
        y = train_batch.y[:, :self.params.PREDICTION_WINDOW]
        edge_index = train_batch.edge_index
        edge_weight = train_batch.edge_attr

        if self.params.FORECASTING:
            main_opt = self.optimizers()
            if self.params.MULTIVARIATE:
                y_predicted= self.forward(x, edge_index, edge_weight)
                #y_predicted = get_only_day_data(y, y_predicted, self.params.DEVICE)
                loss_forecasting = self.mse_f(y_predicted, y)
                loss_forecasting_tot = loss_forecasting
                main_opt.zero_grad()
                self.manual_backward(loss_forecasting_tot)
                main_opt.step()
                self.log('train_loss_FM_power', loss_forecasting, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True, batch_size=1)
                train_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
                self.log('train_MAE_FM', train_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         batch_size=1)

            else:
                y_predicted = self.forward(x, edge_index, edge_weight)
                #y_predicted = get_only_day_data(y, y_predicted, self.params.DEVICE)
                loss_forecasting = self.mse_f(y_predicted, y)
                main_opt.zero_grad()
                self.manual_backward(loss_forecasting)
                main_opt.step()
                self.log('train_loss_FU', loss_forecasting, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True, batch_size=1)
                train_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
                self.log('train_MAE_FU', train_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         batch_size=1)

    def validation_step(self, val_batch, batch_idx):
        # Get data from batches
        if self.params.MULTIVARIATE:
            x = val_batch.x
        else:
            x = val_batch.x[:, :self.params.LAGS]
        y = val_batch.y[:, :self.params.PREDICTION_WINDOW]
        x_power = x[:, :self.params.LAGS]
        edge_index = val_batch.edge_index
        edge_weight = val_batch.edge_attr

        if self.params.FORECASTING:
            if self.params.MULTIVARIATE:
                y_predicted = self.forward(x, edge_index, edge_weight)
                #y_predicted = get_only_day_data(y, y_predicted, self.params.DEVICE)
                loss_forecasting = self.mse_f(y_predicted, y)
                self.log('val_loss_FM_power', loss_forecasting, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True, batch_size=self.params.BATCH_SIZE)
                val_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
                self.log('val_MAE_FM', val_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         batch_size=1)
            else:
                y_predicted = self.forward(x, edge_index, edge_weight)
                #y_predicted = get_only_day_data(y, y_predicted, self.params.DEVICE)
                loss_forecasting = self.mse_f(y_predicted, y)
                self.log('val_loss_FU', loss_forecasting, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True, batch_size=1)
                val_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
                self.log('val_MAE_FU', val_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         batch_size=1)
            if (batch_idx == self.index) and self.params.LOGGER:
                load_prediction_data_F2(x_power, y, y_predicted, self.params, "Validation")
        self.val_batch_list.append(loss_forecasting)

    def test_step(self, test_batch, batch_idx):
        # Get data from batches
        if self.params.MULTIVARIATE:
            x = test_batch.x
        else:
            x = test_batch.x[:, :self.params.LAGS]
        y = test_batch.y[:, :self.params.PREDICTION_WINDOW]
        x_power = x[:, :self.params.LAGS]
        edge_index = test_batch.edge_index
        edge_weight = test_batch.edge_attr

        if self.params.FORECASTING:
            if self.params.MULTIVARIATE:
                y_predicted = self.forward(x, edge_index, edge_weight)
                #y_predicted = get_only_day_data(y, y_predicted, self.params.DEVICE)
                loss_forecasting = self.mse_f(y_predicted, y)
                self.log('test_loss_FM_power', loss_forecasting, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True, batch_size=1)
                test_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
                self.log('test_MAE_FM', test_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         batch_size=1)
            else:
                y_predicted = self.forward(x, edge_index, edge_weight)
                #y_predicted = get_only_day_data(y, y_predicted, self.params.DEVICE)
                loss_forecasting = self.mse_f(y_predicted, y)
                self.log('test_loss_FU', loss_forecasting, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True, batch_size=1)
                test_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
                self.log('test_MAE_FU', test_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         batch_size=1)
            if (batch_idx == self.index) and self.params.LOGGER:
                load_prediction_data_F2(x_power, y, y_predicted, self.params, "Test")

    def configure_optimizers(self):
        main_opt = torch.optim.Adam(list(self.LSTM_Forecasting.parameters()), lr=self.learning_rate)
        return main_opt