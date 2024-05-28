import torch
import torch.nn as nn
import pytorch_lightning as pl


class LSTMForecasting(pl.LightningModule):
    def __init__(self, params):
        super(LSTMForecasting, self).__init__()
        self.params = params

        self.rnn = nn.LSTM(self.params.INPUT_FEATURE_DIMENSION,
                           self.params.HIDDEN_DIMENSION_SINGLE,
                           num_layers=self.params.NUMBER_LSTM_LAYERS,
                           batch_first=True)

        self.linear = torch.nn.Linear(self.params.HIDDEN_DIMENSION_SINGLE * self.params.LAGS,
                                      self.params.OUTPUT_MLP_DIMENSION_FORECASTING)

    def forward(self, x):
        x = torch.reshape(x, (x.size(0), 5, self.params.LAGS))  # batch, feat, seq
        x = torch.transpose(x, 1, 2)  # batch, seq, feat
        h, _ = self.rnn(x)  # batch, seq, hid
        h = h.reshape(h.size(0), -1)
        h = self.linear(h)
        return h

    def __str__(self):
        return "LSTM"
