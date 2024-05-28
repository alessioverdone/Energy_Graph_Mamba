import json

import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


class DatasetCustom(Dataset):
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

            for i in range(0, standardized_target.shape[0] - self.lags - self.params.PREDICTION_WINDOW, self.params.time_series_step)
        ]

        self.targets = [
            np.concatenate((standardized_target[i:i + self.params.PREDICTION_WINDOW, :].T,
                            standardized_temp[i:i + self.params.PREDICTION_WINDOW, :].T,
                            standardized_wind[i:i + self.params.PREDICTION_WINDOW, :].T,
                            standardized_month[i:i + self.params.PREDICTION_WINDOW, :].T,
                            standardized_hour[i:i + self.params.PREDICTION_WINDOW, :].T), axis=-1)

            for i in range(self.lags, standardized_target.shape[0] - self.params.PREDICTION_WINDOW, self.params.time_series_step)
        ]

        if self.params.limit_time_series:
            self.features = self.features[:self.params.max_time_step]
            self.targets = self.targets[:self.params.max_time_step]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def read_dataset(self, lags):
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
        dataset = DatasetCustom(self.params)
        self.num_station = dataset.number_of_station  # len(loader.features[0])

        len_dataset = len(dataset)
        train_ratio = self.params.train_val_and_test_ratio
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
        for _ in self.val_loader: len_val += 1
        for _ in self.test_loader: len_test += 1
        return len_train, len_val, len_test

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
