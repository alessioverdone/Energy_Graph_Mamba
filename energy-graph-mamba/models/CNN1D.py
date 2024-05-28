import torch
import torch.nn as nn
import pytorch_lightning as pl


class CNNForecasting(pl.LightningModule):
    def __init__(self, params):
        super(CNNForecasting, self).__init__()
        self.params = params
        # CONV1
        kernel_size = 5
        in_channels = self.params.INPUT_FEATURE_DIMENSION
        out_channels = 8
        in_features = 24
        out_features = 20
        padding = int((out_features - in_features + kernel_size - 1) / 2)
        self.cnn1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        in_features_pool = 20
        out_features_pool = 18
        kernel_pool = 3
        padding_pool = int((out_features_pool - in_features_pool + kernel_pool - 1) / 2)
        self.maxPool1d = nn.MaxPool1d(kernel_pool, padding=padding_pool, stride=1)
        self.relu = nn.ReLU()

        # #CONV2
        kernel_size = 5
        in_channels = 8
        out_channels = 16
        in_features = 18
        out_features = 16
        padding = int((out_features - in_features + kernel_size - 1) / 2)
        self.cnn2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        # #CONV3
        kernel_size = 3
        in_channels = 16
        out_channels = 24
        in_features = 16
        out_features = 14
        padding = int((out_features - in_features + kernel_size - 1) / 2)
        self.cnn3 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        # #CONV4
        kernel_size = 3
        in_channels = 24
        out_channels = 32
        in_features = 14
        out_features = 12
        padding = int((out_features - in_features + kernel_size - 1) / 2)
        self.cnn4 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        # LINEAR
        self.mlp = nn.Linear(out_channels * out_features, self.params.OUTPUT_MLP_DIMENSION_FORECASTING)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.params.INPUT_FEATURE_DIMENSION, self.params.LAGS))
        h = self.maxPool1d(self.relu(self.cnn1(x)))
        h = self.relu(self.cnn2(h))
        h = self.relu(self.cnn3(h))
        h = self.relu(self.cnn4(h))
        h = h.reshape(h.size(0), -1)  # (1,1920)
        h = self.mlp(h)
        return h

    def __str__(self):
        return "CNN1D"
