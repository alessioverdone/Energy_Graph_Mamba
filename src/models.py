import torch
from torch import nn
import pytorch_lightning as pl
from torch_geometric_temporal.nn.recurrent import GConvLSTM


class GNN_Forecasting(pl.LightningModule):
    """
    Model for doing forecasting on temporal series data
    Functions:
        __init__
        forward
    """

    def __init__(self, params):
        super(GNN_Forecasting, self).__init__()
        self.params = params
        self.recurrent = GConvLSTM(self.params.INPUT_FEATURE_DIMENSION, self.params.FILTERS,
                                   self.params.FILTER_SIZE)
        self.recurrent2 = GConvLSTM(self.params.FILTERS, self.params.FILTERS,
                                    self.params.FILTER_SIZE)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(self.params.INPUT_MLP_DIMENSION, self.params.OUTPUT_MLP_DIMENSION_FORECASTING)

    def forward(self, x, edge_index, edge_weight):
        h_0 = torch.zeros(x.shape[0], self.params.FILTERS).to(x.device)
        c_0 = torch.zeros(x.shape[0], self.params.FILTERS).to(x.device)
        h_1 = torch.zeros(x.shape[0], self.params.FILTERS).to(x.device)
        c_1 = torch.zeros(x.shape[0], self.params.FILTERS).to(x.device)
        x = torch.reshape(x, (-1, 5, self.params.LAGS))
        for i in range(self.params.LAGS):
            x_t = x[:, :, i]
            h_0, c_0 = self.recurrent(x_t, edge_index, edge_weight, H=h_0, C=c_0)
            h_0 = self.relu(h_0)
            h_1, c_1 = self.recurrent2(h_0, edge_index, edge_weight, H=h_1, C=c_1)
            h_1 = self.tanh(h_1)
        # for i in range(self.params.LAGS):
        #     x_t = x[:, :, i]
        #     h_0, c_0 = self.recurrent(x_t, edge_index, edge_weight, H=h_0, C=c_0)
        #     h_0 = self.relu(h_0)
        #
        # for i in range(self.params.LAGS):
        #     h_1, c_1 = self.recurrent2(h_0, edge_index, edge_weight, H=h_1, C=c_1)
        #     h_1 = self.tanh(h_1)
        h = self.linear(h_1)
        return h


class LSTM_Forecasting(pl.LightningModule):
    def __init__(self, params):
        super(LSTM_Forecasting, self).__init__()
        self.params = params

        self.rnn = nn.LSTM(self.params.INPUT_FEATURE_DIMENSION,
                           self.params.HIDDEN_DIMENSION_SINGLE,
                           num_layers=self.params.NUMBER_LSTM_LAYERS,
                           batch_first=True)

        self.linear = torch.nn.Linear(self.params.HIDDEN_DIMENSION_SINGLE * self.params.LAGS,
                                      self.params.OUTPUT_MLP_DIMENSION_FORECASTING)

    def forward(self, x):
        if self.params.MULTIVARIATE:
            x = torch.reshape(x, (x.size(0), 5, self.params.LAGS))  # batch, feat, seq
        else:
            x = torch.unsqueeze(x, dim =1)
        x = torch.transpose(x, 1, 2)  # batch, seq, feat
        h, _ = self.rnn(x)  # batch, seq, hid
        h = h.reshape(h.size(0), -1)
        h = self.linear(h)
        return h


class CNN_Forecasting(pl.LightningModule):
    def __init__(self, params):
        super(CNN_Forecasting, self).__init__()
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
        self.mlp = nn.Linear(out_channels*out_features, self.params.OUTPUT_MLP_DIMENSION_FORECASTING)

    def forward(self, x):
        if self.params.MULTIVARIATE:
            x = torch.reshape(x, (-1, self.params.INPUT_FEATURE_DIMENSION, self.params.LAGS))
        else:
            x = torch.unsqueeze(x, dim=1)
        h = self.maxPool1d(self.relu(self.cnn1(x)))
        h = self.relu(self.cnn2(h))
        h = self.relu(self.cnn3(h))
        h = self.relu(self.cnn4(h))
        h = h.reshape(h.size(0), -1)  # (1,1920)
        h = self.mlp(h)
        return h


