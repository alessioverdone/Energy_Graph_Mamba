import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import mean_absolute_error as MAE

from utils import load_prediction_data, get_model, get_index_for_visualization

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class TrainingModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.index = 0
        self.val_min_length = None
        self.learning_rate = self.params.LR
        self.automatic_optimization = False
        self.val_loss_during_training = []
        self.current_val_loss = 10000
        self.val_batch_list = []
        self.model = get_model(params)
        self.loss = nn.MSELoss()

    def on_train_epoch_start(self):  # on_epoch_start
        # Define an index for the generation of a random prediction at validation step
        self.index = get_index_for_visualization(self.params)

        # Updating LR of a factor of DECAY_LR every LR_ITER epochs
        if self.current_epoch % self.params.LR_ITER == 0 and self.current_epoch > 0:
            print('Updating learning rate at epoch: ', self.current_epoch)
            main_opt = self.optimizers()
            self.learning_rate = self.learning_rate * self.params.DECAY_LR
            for group in main_opt.param_groups:
                group["lr"] = self.learning_rate
        self.log('lr', self.learning_rate, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        mean_val_loss = torch.Tensor(self.val_batch_list).mean()
        self.val_loss_during_training.append(mean_val_loss)
        self.val_batch_list = list()

    def forward(self, batch, edge_index, edge_weight):
        if str(self.model) == "GConvLSTM":
            output = self.model(batch, edge_index, edge_weight)
        elif str(self.model) == "LSTM" or str(self.model) == "CNN1D":
            output = self.model(batch)
        else:
            raise Exception("Define model name!")
        return output

    def training_step(self, train_batch, batch_idx):
        # Get data from batches
        x, y, edge_index, edge_weight = (train_batch.x,
                                         train_batch.y[:, :self.params.PREDICTION_WINDOW],
                                         train_batch.edge_index,
                                         train_batch.edge_attr)

        # Optimize
        main_opt = self.optimizers()
        y_predicted = self.forward(x, edge_index, edge_weight)
        loss_forecasting = self.loss(y_predicted, y)
        main_opt.zero_grad()
        self.manual_backward(loss_forecasting)
        main_opt.step()

        # Log metrics
        train_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        self.log('train_loss', loss_forecasting, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.params.BATCH_SIZE)
        self.log('train_MAE', train_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.params.BATCH_SIZE)

    def validation_step(self, val_batch, batch_idx):
        # Get data from batches
        x, y, edge_index, edge_weight = (val_batch.x,
                                         val_batch.y[:, :self.params.PREDICTION_WINDOW],
                                         val_batch.edge_index,
                                         val_batch.edge_attr)
        x_power = x[:, :self.params.LAGS]

        # Predict
        y_predicted = self.forward(x, edge_index, edge_weight)
        loss_forecasting = self.loss(y_predicted, y)

        # Log Metrics
        val_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        self.log('val_loss', loss_forecasting, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.params.BATCH_SIZE)
        self.log('val_MAE', val_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.params.BATCH_SIZE)

        # Load validation images
        if (batch_idx == self.index) and self.params.USE_LOGGER:
            load_prediction_data(x_power, y, y_predicted, self.params, "Validation")
        self.val_batch_list.append(loss_forecasting)

    def test_step(self, test_batch, batch_idx):
        # Get data from batches
        x, y, edge_index, edge_weight = (test_batch.x,
                                         test_batch.y[:, :self.params.PREDICTION_WINDOW],
                                         test_batch.edge_index,
                                         test_batch.edge_attr)
        x_power = x[:, :self.params.LAGS]

        # Predict
        y_predicted = self.forward(x, edge_index, edge_weight)
        loss_forecasting = self.loss(y_predicted, y)

        # Log Metrics
        test_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        self.log('test_loss', loss_forecasting, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.params.BATCH_SIZE)
        self.log('test_MAE', test_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.params.BATCH_SIZE)

        # Load validation images
        if (batch_idx == self.index) and self.params.USE_LOGGER:
            load_prediction_data(x_power, y, y_predicted, self.params, "Test")

    def configure_optimizers(self):
        main_opt = torch.optim.Adam(list(self.model.parameters()), lr=self.learning_rate)
        return main_opt
