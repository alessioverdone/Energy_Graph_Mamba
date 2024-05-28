import torch
import pytorch_lightning as pl
from torch_geometric_temporal.nn.recurrent import GConvLSTM


class GNNForecasting(pl.LightningModule):
    def __init__(self, params):
        super(GNNForecasting, self).__init__()
        self.params = params
        self.recurrent = GConvLSTM(self.params.INPUT_FEATURE_DIMENSION,
                                   self.params.FILTERS,
                                   self.params.FILTER_SIZE)
        self.recurrent2 = GConvLSTM(self.params.FILTERS,
                                    self.params.FILTERS,
                                    self.params.FILTER_SIZE)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(self.params.INPUT_MLP_DIMENSION, self.params.OUTPUT_MLP_DIMENSION_FORECASTING)

    def forward(self, x, edge_index, edge_weight):
        h_0 = torch.zeros(x.shape[0], self.params.FILTERS).to(x.device)
        c_0 = torch.zeros(x.shape[0], self.params.FILTERS).to(x.device)
        h_1 = torch.zeros(x.shape[0], self.params.FILTERS).to(x.device)
        c_1 = torch.zeros(x.shape[0], self.params.FILTERS).to(x.device)
        # x has shape: [Batch x #Nodes, #Features x #Lags]
        x = torch.reshape(x, (-1, 5, self.params.LAGS))
        # x has shape: [Batch x #Nodes, #Features, #Lags]
        for i in range(self.params.LAGS):
            x_t = x[:, :, i]
            h_0, c_0 = self.recurrent(x_t, edge_index, edge_weight, H=h_0, C=c_0)
            h_0 = self.relu(h_0)
            h_1, c_1 = self.recurrent2(h_0, edge_index, edge_weight, H=h_1, C=c_1)
            h_1 = self.tanh(h_1)

        h = self.linear(h_1)
        return h

    def __str__(self):
        return "GConvLSTM"
