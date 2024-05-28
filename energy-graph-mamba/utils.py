import numpy as np
import torch
import random
import wandb
import io
import PIL
import matplotlib
import matplotlib.pyplot as plt

from models.CNN1D import CNNForecasting
from models.GConvLSTM import GNNForecasting
from models.LSTM import LSTMForecasting


def getVariablesClass(inst):
    var = []
    cls = inst.__class__
    for v in cls.__dict__:
        if not callable(getattr(cls, v)):
            var.append(v)
    return var


def get_hyperparams_dict(p):
    params_list = getVariablesClass(p)
    d = dict()
    for pos, var in enumerate(params_list):
        d[var] = getattr(p, var)
    return d


def get_only_day_data(y, h, device):
    """
    Inputs:
        y : data to rpedict
    Outputs:
        night_mask : 1=day, 0=night
    """
    y.to(device)
    h.to(device)
    dim_y = y.shape
    ones = torch.ones(dim_y).to(device)
    zeros = torch.zeros(dim_y).to(device)
    mask = torch.where(y == 0.0, zeros, ones)
    h = mask * h

    return h


def buffer_plot_and_get(fig):
    """
    Util function for visualization output
    """
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)


def load_prediction_data(x, y_f, h_f, params, phase):
    """
    Inputs:
        x : data batch corrupted
        h : predictions
        y : targets
    Outputs:
        Load images on wandb logger
    """

    index = random.randint(0, params.NUM_STATION - 1)
    features_f = x[index, :]
    predictions_f = h_f[index, :]
    targets_f = y_f[index, :]
    image_plot, plt_plot = visualization_output(features_f,
                                                targets_f,
                                                predictions_f,
                                                params,
                                                station_name=str(index))
    wandb.log({phase + "_images": wandb.Image(image_plot)})


def visualization_output(x, y_f, h_f, params, station_name):
    """
    Create images from features, preditions and targets data for data imputation task
    """

    params_img = {
        'axes.labelsize': 10,
        'font.family': 'Times New Roman',
        'font.size': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'text.usetex': False,
        'figure.figsize': [4.7, 3.2],
        'lines.linewidth': 1.2
    }

    matplotlib.rcParams.update(params_img)
    matplotlib.rc('pdf', fonttype=42)
    # matplotlib.font_manager._rebuild()

    x_axis_feature = np.arange(params.LAGS)
    x_axis_target = np.arange(params.LAGS - 1, params.LAGS + params.PREDICTION_WINDOW)
    target_sequence = torch.cat((x[-1].unsqueeze(dim=0).cpu(), y_f.cpu()))
    predicted_sequence = torch.cat((x[-1].unsqueeze(dim=0).cpu(), h_f.cpu()))

    fig, ax = plt.subplots()
    ax.yaxis.grid(linewidth=0.5, alpha=0.3)

    plt.plot(x_axis_feature, x.cpu(), linestyle='-', color='b', label="S_x")  # Input sequence"
    plt.plot(x_axis_target, target_sequence.cpu(), linestyle='--', color="b", label="S_y")  # Target sequence
    plt.plot(x_axis_target, predicted_sequence.cpu(), linestyle='-', color="r", label="S_h")  # Predicted sequence
    plt.plot([x_axis_feature[-1], x_axis_feature[-1]], [-0.2, 1], linestyle='-.',
             color='k')

    plt.xlabel('Hours')
    plt.ylabel('Normalized power')
    ax.legend()
    plt.tight_layout()

    if params.SAVE_IMGS:
        fig.savefig('imgs/' + str(random.randint(1, 1000000)) + '_prova.pdf', bbox_inches='tight', pad_inches=0)

    plt.title(station_name)
    pil_image = buffer_plot_and_get(fig)
    return pil_image, plt


def get_run_name(db, input_ws, output_ws, params):
    name_run = "%s_%s_%s_%s_%s" % (params.MODEL, 'FM', db, input_ws, output_ws)
    return name_run


def get_model(p):
    if p.MODEL == "GConvLSTM":
        model = GNNForecasting(p)
    elif p.MODEL == "LSTM":
        model = LSTMForecasting(p)
    elif p.MODEL == "CNN1D":
        model = CNNForecasting(p)
    else:
        raise Exception("Choose a correct model")
    return model


def get_index_for_visualization(p):
    val_min_length = p.LIMIT_VAL_BATCHES * p.LEN_VAL
    index = np.random.randint(0, val_min_length - 1)
    return index
