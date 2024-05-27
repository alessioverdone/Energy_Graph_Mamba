import torch
import random
import wandb
import io
import PIL
import torch.autograd as autograd
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from pylab import rcParams
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from palettable.cartocolors.sequential import SunsetDark_6


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


def degrade_dataset(X, missingness, rand, v):
    """
    Inputs:
        X : dataset to corrupt
        missingness : % of data to eliminate[0,1]
        rand : random state
        v : replace with = 'zero' or 'nan'
      Outputs:
        corrupted Dataset
        binary mask
    """
    x_temp = X.clone()
    X_1d = x_temp.flatten()  # X.shape = #lags x #station
    n = len(X_1d)
    # mask_1d = torch.ones(n)
    mask_1d = torch.zeros(n)

    corrupt_ids = random.sample(range(n), int(missingness * n))
    for i in corrupt_ids:
        X_1d[i] = v
        # mask_1d[i] = 0
        mask_1d[i] = 1

    cX = X_1d.reshape(X.shape)
    mask = mask_1d.reshape(X.shape)
    mask = mask.byte()

    return cX, mask


def check_mask(x, device, v=-1):
    """
    Inputs:
        x : data batch corrupted
    Outputs:
        mask : 1=corrupted, 0=original # nel paper fa l'opposto
    """
    x.to(device)
    if x.dim() == 1:
        dim_x = x.shape[0]
    else:
        dim_x = x.shape[1]
    ones = torch.ones(dim_x).to(device)
    zeros = torch.zeros(dim_x).to(device)
    mask = torch.where(x != v, zeros, ones)
    return mask


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


def visualization_output_imputation(features, targets, predictions, predictions_rec, params, station_name):
    """
    Create images from features, preditions and targets data for data imputation task
    """
    x_axis_feature = np.arange(params.LAGS)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(x_axis_feature, features.cpu(), "-g", label="features_" + station_name)
    ax.plot(x_axis_feature, targets.cpu(), "-b", label="target_" + station_name)
    ax.plot(x_axis_feature, predictions_rec.cpu(), "-r", label="predicted_rec_" + station_name)
    ax.plot(x_axis_feature, predictions.cpu(), "-y", label="predicted_" + station_name)
    plt.title(station_name)
    pil_image = buffer_plot_and_get(fig)
    return pil_image, plt


def visualization_output_imputation_F(x, y_f, h_f, params, station_name):
    """
    Create images from features, preditions and targets data for data imputation task
    """
    x_axis_feature = np.arange(params.LAGS)
    x_axis_target = np.arange(params.LAGS - 1, params.LAGS + params.PREDICTION_WINDOW)
    target_sequence = torch.cat((x[-1].unsqueeze(dim=0).cpu(), y_f.cpu()))
    predicted_sequence = torch.cat((x[-1].unsqueeze(dim=0).cpu(), h_f.cpu()))
    fig, ax = plt.subplots(figsize=(25, 15))

    ax.plot(x_axis_feature, x.cpu(), linestyle='-', color='b', label="y_i_" + station_name)
    ax.plot(x_axis_target, target_sequence.cpu(), linestyle='--', color="b", label="y_f_" + station_name)
    ax.plot(x_axis_target, predicted_sequence.cpu(), linestyle='-', color="r", label="h_f_" + station_name)
    ax.plot([x_axis_feature[-1], x_axis_feature[-1]], [-0.2, 1], linestyle='-.', color='k',
            label="limit_" + station_name)

    plt.title(station_name)
    plt.xlabel("Hours")
    # plt.ylabel("Output")
    pil_image = buffer_plot_and_get(fig)

    return pil_image, plt


def visualization_output_imputation_IF(x_c, y_i, h_i, h_i_rec, y_f, h_f, params, station_name):
    """
    Create images from features, preditions and targets data for data imputation task
    """
    x_axis_feature = np.arange(params.LAGS)
    x_axis_target = np.arange(params.LAGS - 1, params.LAGS + params.PREDICTION_WINDOW)
    target_sequence = torch.cat((y_i[-1].unsqueeze(dim=0).cpu(), y_f.cpu()))
    predicted_sequence = torch.cat((y_i[-1].unsqueeze(dim=0).cpu(), h_f.cpu()))
    h_i = torch.cat((h_i.cpu(), y_f[0].unsqueeze(dim=0).cpu()))
    # plt.stem(x_axis_feature, np.array(x_c))
    fig, ax = plt.subplots(figsize=(25, 15))

    # Combination of style #1

    # ax.plot(x_axis_feature, x_c.cpu(), 'o-', color='g',  label="x_c_" + station_name) #'o-',
    # ax.plot(x_axis_feature, y_i.cpu(), 'o-', color='b', label="y_i_" + station_name)
    # ax.plot(x_axis_feature_1, h_i.cpu(), 'o-', color="y", label="h_i_" + station_name)
    # ax.plot(x_axis_target, target_sequence.cpu(), 'o-', color="b", label="y_f_" + station_name)
    # ax.plot(x_axis_target, predicted_sequence.cpu(), 'o-', color="r", label="h_f_" + station_name)
    # ax.plot(x_axis_feature, h_i_rec.cpu(), 'o-', color="r", label="h_i_rec_" + station_name)

    # Combination of style #2
    # ax.plot(x_axis_feature, x_c.cpu(), linestyle='--', color='g',  label="x_c_" + station_name) #'o-', 'r*'
    mask = check_mask(x_c.cpu(), 'cpu').byte()
    x_axis_feature_tensor = torch.tensor(x_axis_feature)
    x_axis_corrupted = x_axis_feature_tensor[mask]
    ax.plot(x_axis_feature, y_i.cpu(), linestyle='--', color='b', label="y_i_" + station_name)
    # ax.plot(x_axis_feature_1, h_i.cpu(), linestyle='-.', color="y", label="h_i_" + station_name) # all x predicted
    ax.plot(x_axis_target, target_sequence.cpu(), linestyle='--', color="b", label="y_f_" + station_name)
    ax.plot(x_axis_target, predicted_sequence.cpu(), linestyle='-', color="r", label="h_f_" + station_name)
    ax.plot(x_axis_feature, h_i_rec.cpu(), linestyle='-', color="r", label="h_i_rec_" + station_name)
    ax.plot(x_axis_corrupted, np.ones(len(x_axis_corrupted)) * -0.001, 'kP', label="x_c_" + station_name)  # 'o-',
    ax.plot([x_axis_feature[-1], x_axis_feature[-1]], [-0.2, 1], linestyle='-.', color='k',
            label="limit_" + station_name)

    plt.title(station_name)
    plt.xlabel("Hours")
    # plt.ylabel("Output")
    pil_image = buffer_plot_and_get(fig)
    return pil_image, plt


def load_prediction_data(x, h, y, params, mask, phase):
    """
    Inputs:
        x : data batch corrupted
        h : predictions
        y : targets
    Outputs:
        Load images on wandb logger
    """

    index = random.randint(0, params.NUM_STATION - 1)
    mask = (mask[index, :]).float()
    features = x[index, :]
    predictions = h[index, :]
    targets = y[index, :]
    predictions_reconstructed = torch.mul(mask, predictions) + torch.mul(1 - mask, features)
    image_plot, plt_plot = visualization_output_imputation(features, targets, predictions, predictions_reconstructed,
                                                           params,
                                                           station_name=str(index))
    wandb.log({phase + "_plot": plt_plot})
    wandb.log({phase + "_images": wandb.Image(image_plot)})


def load_prediction_data_IF(x_c, h_i, y_i, y_f, h_f, params, mask, phase):
    """
    Inputs:
        x : data batch corrupted
        h : predictions
        y : targets
    Outputs:
        Load images on wandb logger
    """

    index = random.randint(0, params.NUM_STATION - 1)
    mask = (mask[index, :]).float()
    features_corrupted = x_c[index, :]
    predictions_i = h_i[index, :]
    targets_i = y_i[index, :]
    predictions_f = h_f[index, :]
    targets_f = y_f[index, :]
    predictions_reconstructed_i = torch.mul(mask, predictions_i) + torch.mul(1 - mask, targets_i)
    image_plot, plt_plot = visualization_output_imputation_IF(features_corrupted, targets_i, predictions_i,
                                                              predictions_reconstructed_i,
                                                              targets_f,
                                                              predictions_f,
                                                              params,
                                                              station_name=str(index))
    wandb.log({phase + "_plot": plt_plot})
    wandb.log({phase + "_images": wandb.Image(image_plot)})


def load_prediction_data_F(x, y_f, h_f, params, phase):
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
    image_plot, plt_plot = visualization_output_imputation_F(features_f,
                                                             targets_f,
                                                             predictions_f,
                                                             params,
                                                             station_name=str(index))
    wandb.log({phase + "_plot": plt_plot})
    wandb.log({phase + "_images": wandb.Image(image_plot)})


def load_prediction_data_F2(x, y_f, h_f, params, phase):
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
    image_plot, plt_plot = visualization_output_imputation_F2(features_f,
                                                              targets_f,
                                                              predictions_f,
                                                              params,
                                                              station_name=str(index))
    # wandb.log({phase + "_plot": plt_plot})
    wandb.log({phase + "_images": wandb.Image(image_plot)})


def visualization_output_imputation_F2(x, y_f, h_f, params, station_name):
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

    rcParams.update(params_img)
    matplotlib.rc('pdf', fonttype=42)
    # matplotlib.font_manager._rebuild()

    x_axis_feature = np.arange(params.LAGS)
    x_axis_target = np.arange(params.LAGS - 1, params.LAGS + params.PREDICTION_WINDOW)
    target_sequence = torch.cat((x[-1].unsqueeze(dim=0).cpu(), y_f.cpu()))
    predicted_sequence = torch.cat((x[-1].unsqueeze(dim=0).cpu(), h_f.cpu()))

    colors = SunsetDark_6.mpl_colors
    fig, ax = plt.subplots()
    ax.yaxis.grid(linewidth=0.5, alpha=0.3)

    # plt.plot(np.ones(10), color=colors[0], label='Prova')
    plt.plot(x_axis_feature, x.cpu(), linestyle='-', color='b', label="S_x")  # Input sequence"
    plt.plot(x_axis_target, target_sequence.cpu(), linestyle='--', color="b", label="S_y")  # Target sequence
    plt.plot(x_axis_target, predicted_sequence.cpu(), linestyle='-', color="r", label="S_h")  # Predicted sequence
    plt.plot([x_axis_feature[-1], x_axis_feature[-1]], [-0.2, 1], linestyle='-.',
             color='k')  # , label="Current timestep")

    # plt.xlim(0, 9)
    # plt.ylim(0, 2)
    plt.xlabel('Hours')
    plt.ylabel('Normalized power')
    ax.legend()
    plt.tight_layout()
    # plt.show()
    id = random.randint(1, 1000000)
    if params.SAVE_IMGS:
        fig.savefig('imgs/' + str(id) + '_prova.pdf', bbox_inches='tight', pad_inches=0)

    # fig, ax = plt.subplots(figsize=(25, 15))

    # ax.plot(x_axis_feature, x.cpu(), linestyle='-', color='b', label="y_i_" + station_name)
    # ax.plot(x_axis_target, target_sequence.cpu(), linestyle='--', color="b", label="y_f_" + station_name)
    # ax.plot(x_axis_target, predicted_sequence.cpu(), linestyle='-', color="r", label="h_f_" + station_name)
    # ax.plot([x_axis_feature[-1], x_axis_feature[-1]], [-0.2, 1], linestyle='-.', color='k',
    #         label="limit_" + station_name)

    plt.title(station_name)
    # plt.xlabel("Hours")
    # plt.ylabel("Output")
    pil_image = buffer_plot_and_get(fig)

    return pil_image, plt


def hard_gradient_penalty(net, real_data, fake_data, device):
    mask = torch.FloatTensor(real_data.shape).to(device).uniform_() > 0.5
    inv_mask = ~mask
    mask, inv_mask = mask.float(), inv_mask.float()

    interpolates = mask * real_data + inv_mask * fake_data
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    c_interpolates = net(interpolates)

    gradients = autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(c_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = (gradients.norm(2, dim=1) - 1).pow(2).mean()
    return gp


# Cumulative loss display
def loss_used(params):
    loss_list = []

    # if params.IMPUTATION:
    #     loss_list.append('test_loss_imputation')
    #     loss_list.append('test_MAE_imputation')
    #     if params.FORECASTING:
    #         if params.MULTIVARIATE:
    #             loss_list.append('test_loss_IFM_power')
    #             loss_list.append('test_loss_IFM_temp')
    #             loss_list.append('test_loss_IFM_wind')
    #             loss_list.append('test_MAE_IFM')
    #         else:
    #             loss_list.append('test_loss_IFU')
    #             loss_list.append('test_MAE_IFU')
    # else:
    if params.FORECASTING:
        if params.MULTIVARIATE:
            loss_list.append('test_loss_FM_power')
            loss_list.append('test_MAE_FM')
            # loss_list.append('test_loss_FM_temp')
            # loss_list.append('test_loss_FM_wind')
        else:
            loss_list.append('test_loss_FU')
            loss_list.append('test_MAE_FU')

    return loss_list


def get_value_of_same_loss(list_of_dict, name):
    values = []
    for d in list_of_dict:
        values.append(d[name])
    return np.array(values)


def print_runs_results(params, RUNS, results):
    loss_names = loss_used(params)
    for name in loss_names:
        loss_values = get_value_of_same_loss(results, name)
        mean = loss_values.mean()
        std = loss_values.std()
        print("----", name, "----")
        print("Mean: ", mean, "Std: ", std)
        print("EXC_Mean: ", str(mean).replace('.', ','), "EXC_Std: ", str(std).replace('.', ','), '\n')
        # print(loss_values)


# Shuffle dataset
def temporal_signal_split_and_shuffle(data_iterator, shuffle=True, train_ratio: float = 0.8):
    if shuffle:
        train_snapshots = int(train_ratio * data_iterator.snapshot_count)
        total_lenght = data_iterator.snapshot_count
        feature_index = np.arange(total_lenght, dtype=int)
        np.random.shuffle(feature_index)
        train_index = feature_index[0:train_snapshots].tolist()
        test_index = feature_index[train_snapshots:].tolist()

        if type(data_iterator) == StaticGraphTemporalSignal:
            train_features = np.array(data_iterator.features)[train_index]
            train_targets = np.array(data_iterator.targets)[train_index]
            test_features = np.array(data_iterator.features)[test_index]
            test_targets = np.array(data_iterator.targets)[test_index]
            train_iterator = StaticGraphTemporalSignal(
                data_iterator.edge_index,
                data_iterator.edge_weight,
                train_features,
                train_targets,
            )

            test_iterator = StaticGraphTemporalSignal(
                data_iterator.edge_index,
                data_iterator.edge_weight,
                test_features,
                test_targets,
            )

    else:
        train_snapshots = int(train_ratio * data_iterator.snapshot_count)
        if type(data_iterator) == StaticGraphTemporalSignal:
            train_iterator = StaticGraphTemporalSignal(
                data_iterator.edge_index,
                data_iterator.edge_weight,
                data_iterator.features[0:train_snapshots],
                data_iterator.targets[0:train_snapshots],
            )

            test_iterator = StaticGraphTemporalSignal(
                data_iterator.edge_index,
                data_iterator.edge_weight,
                data_iterator.features[train_snapshots:],
                data_iterator.targets[train_snapshots:],
            )

    return train_iterator, test_iterator


# Define run name
def get_run_name(db, input_ws, output_ws, params):
    # Test
    test = '-'
    if params.FORECASTING:
        test = 'F'
        if params.MULTIVARIATE:
            test += 'M'

    name_run = "%s_%s_%s_%s_%s" % (params.GNN_MODEL, test, db, input_ws, output_ws)
    return name_run



def plot_and_save_loss(loss, total_time, params):
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

    rcParams.update(params_img)
    matplotlib.rc('pdf', fonttype=42)
    # matplotlib.font_manager._rebuild()

    colors = SunsetDark_6.mpl_colors
    fig, ax = plt.subplots()
    ax.yaxis.grid(linewidth=0.5, alpha=0.3)

    x_axis = np.arange(0,len(loss)*5, 5)
    plt.plot(x_axis, torch.tensor(loss).cpu(), linestyle='-', color='b', label="Loss")  # Input sequence"
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    ax.legend()
    plt.tight_layout()
    id = random.randint(1, 1000000)
    fig.savefig('imgs/' + str(id) + '_prova.pdf', bbox_inches='tight', pad_inches=0)
    plt.title("Validation loss during training")

    fig2, ax2 = plt.subplots()
    ax2.yaxis.grid(linewidth=0.5, alpha=0.3)
    time_per_epoch = total_time/params.EPOCHS
    curr_step = 0
    x_axis_time = [0]
    for i in range(len(loss) -1 ):
        curr_step += 5*time_per_epoch
        x_axis_time.append(curr_step)
    x_axis_time = np.array(x_axis_time)
    plt.plot(x_axis_time, torch.tensor(loss).cpu(), linestyle='-', color='b', label="Loss")  # Input sequence"
    plt.xlabel('Time')
    plt.ylabel('MSE Loss')
    ax2.legend()
    plt.tight_layout()
    id = random.randint(1, 1000000)
    fig2.savefig('imgs/' + str(id) + '_prova.pdf', bbox_inches='tight', pad_inches=0)
    plt.title("Validation loss during training")
    return x_axis_time
