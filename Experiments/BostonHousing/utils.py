from sklearn.datasets import load_boston
from pandas import Series,DataFrame
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _get_index_train_test_path(split_num, train = True):

    _DATA_DIRECTORY_PATH = './data/'

    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
    else:
        return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt"

def load_data():

    boston = load_boston()
    X = boston.data
    Y = boston.target
    return X, Y

def get_data_splited(split, X, Y):
    # We load the indexes of the training and test sets
    print('Loading file: ' + _get_index_train_test_path(split, train=True))
    print('Loading file: ' + _get_index_train_test_path(split, train=False))
    index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

    X_train = X[[int(i) for i in index_train.tolist()]]
    y_train = Y[[int(i) for i in index_train.tolist()]]

    X_test = X[[int(i) for i in index_test.tolist()]]
    y_test = Y[[int(i) for i in index_test.tolist()]]

    y_train = y_train.reshape([y_train.shape[0], 1])
    y_test = y_test.reshape([y_test.shape[0], 1])

    num_training_examples = int(0.8 * X_train.shape[0])

    X_val = X_train[num_training_examples:, :]
    y_val = y_train[num_training_examples:, :]

    X_train = X_train[0:num_training_examples, :]
    y_train = y_train[0:num_training_examples, :]

    x_means, x_stds = X_train.mean(axis=0), X_train.var(axis=0) ** 0.5
    y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0) ** 0.5

    X_train = (X_train - x_means) / x_stds
    y_train = (y_train - y_means) / y_stds

    X_val = (X_val - x_means) / x_stds
    y_val = (y_val - y_means) / y_stds

    X_test = (X_test - x_means) / x_stds
    y_test = (y_test - y_means) / y_stds

    return X_train, y_train, X_val, y_val, X_test, y_test, y_stds

def get_dataset(X_train, y_train, X_val, y_val, X_test, y_test):

    x_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    print(x_train.size(), y_train.size())
    trainset = torch.utils.data.TensorDataset(x_train, y_train)

    x_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float()
    print(x_val.size(), y_val.size())
    valset = torch.utils.data.TensorDataset(x_val, y_val)

    x_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    print(x_test.size(), y_test.size())
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    return trainset, valset, testset

def get_dataloader(trainset, valset, testset, use_cuda, batch_size, worker=True):
    if worker:
        if use_cuda:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True, pin_memory=True,
                                                      num_workers=3)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                    shuffle=False, pin_memory=True,
                                                    num_workers=3)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=False, pin_memory=True,
                                                     num_workers=3)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True, pin_memory=False,
                                                      num_workers=3)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                    shuffle=False, pin_memory=False,
                                                    num_workers=3)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=False, pin_memory=False,
                                                     num_workers=3)
    else:
        if use_cuda:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                    shuffle=False)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=False)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                    shuffle=False)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=False)
    return trainloader, valloader, testloader

def get_dataloader_sample(trainset, valset, testset, use_cuda, batch_size, sampler, worker=True):
    if worker:
        if use_cuda:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=False, pin_memory=True,
                                                      num_workers=3, sampler=sampler)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                    shuffle=False, pin_memory=True,
                                                    num_workers=3)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=False, pin_memory=True,
                                                     num_workers=3)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=False, pin_memory=False,
                                                      num_workers=3, sampler=sampler)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                    shuffle=False, pin_memory=False,
                                                    num_workers=3)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=False, pin_memory=False,
                                                     num_workers=3)
    else:
        if use_cuda:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=False, sampler=sampler)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                    shuffle=False)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=False)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=False, sampler=sampler)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                    shuffle=False)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=False)
    return trainloader, valloader, testloader

def store_results(file, results):
    with open(file, "a") as myfile:
        for str in results:
            myfile.write(str)

def plot_pred_cost(pred_cost_train, nb_epochs, nb_its_dev, cost_dev, results_dir_split):

    textsize = 15
    marker = 5

    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(pred_cost_train, 'r--')
    ax1.plot(range(0, nb_epochs, nb_its_dev), cost_dev[::nb_its_dev], 'b-')
    ax1.set_ylabel('MSE loss')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['train cost', 'val cost'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('Regression costs')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir_split + '/pred_cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_kl_cost(kl_cost_train, results_dir_split):

    textsize = 15
    marker = 5

    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.plot(kl_cost_train, 'r')
    ax1.set_ylabel('nats?')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['train cost', 'val cost'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('DKL (per sample)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir_split + '/KL_cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_rmse(nb_epochs, nb_its_dev, rmse_train, rmse_dev, results_dir_split):

    textsize = 15
    marker = 5

    plt.figure(dpi=100)
    fig2, ax2 = plt.subplots()
    ax2.set_ylabel('% rmse')
    ax2.semilogy(range(0, nb_epochs, nb_its_dev), 100 * rmse_dev[::nb_its_dev], 'b-')
    if rmse_train is not None:
        ax2.semilogy(100 * rmse_train, 'r--')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    lgd = plt.legend(['val rmse', 'train rmse'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir_split + '/rmse.png', bbox_extra_artists=(lgd,), box_inches='tight')

def plot_uncertainty_noise(means, noise, total_unc, y_test, results_dir_split):

    textsize = 15
    marker = 5

    means = means.reshape((means.shape[0],))
    noise = noise.reshape((noise.shape[0],))
    total_unc_1 = total_unc[0].reshape((total_unc[0].shape[0],))
    total_unc_2 = total_unc[1].reshape((total_unc[1].shape[0],))
    total_unc_3 = total_unc[2].reshape((total_unc[2].shape[0],))

    c = ['#1f77b4', '#ff7f0e']
    ind = np.arange(0, len(y_test))
    plt.figure()
    fig, ax1 = plt.subplots()
    plt.scatter(ind, y_test, color='black', alpha=0.5)
    ax1.plot(ind, means, 'r')
    plt.fill_between(ind, means - total_unc_3, means + total_unc_3,
                     alpha=0.25, label='99.7% Confidence')
    plt.fill_between(ind, means - total_unc_2, means + total_unc_2,
                     alpha=0.25, label='95% Confidence')
    plt.fill_between(ind, means - total_unc_1, means + total_unc_1,
                     alpha=0.25, label='68% Confidence')
    plt.fill_between(ind, means - noise, means + noise,
                     alpha=0.25, label='Noise')
    ax1.set_ylabel('prediction')
    plt.xlabel('test points')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['prediction mean'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('Uncertainty')

    plt.savefig(results_dir_split + '/uncertainty.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_uncertainty(means, stds, y_test, results_dir_split):

    textsize = 15
    marker = 5

    means = means.reshape((means.shape[0],))
    stds = stds.reshape((stds.shape[0],))

    c = ['#1f77b4', '#ff7f0e']
    ind = np.arange(0, len(y_test))
    plt.figure()
    fig, ax1 = plt.subplots()
    plt.scatter(ind, y_test, color='black', alpha=0.5)
    ax1.plot(ind, means, 'r')
    plt.fill_between(ind, means - 3 * stds, means + 3 * stds,
                     alpha=0.25, label='99.7% Confidence')
    plt.fill_between(ind, means - 2 * stds, means + 2 * stds,
                     alpha=0.25, label='95% Confidence')
    plt.fill_between(ind, means - stds, means + stds,
                     alpha=0.25, label='68% Confidence')
    ax1.set_ylabel('prediction')
    plt.xlabel('test points')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['prediction mean'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('Uncertainty')

    plt.savefig(results_dir_split + '/uncertainty.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def store_all_results(results, path):
    results_order_rmse = sorted(results.items(), key=lambda x: x[1][0], reverse=False)
    for i in range(len(results_order_rmse)):
        with open(path+'/results_rmse.txt', 'a') as f:
            f.write(str(results_order_rmse[i][0]) + ' RMSE: %f +- %f (stddev) +- %f (std error) PICP %f MPIW %f'
                    % (results_order_rmse[i][1][0], results_order_rmse[i][1][1], results_order_rmse[i][1][2], results_order_rmse[i][1][3],results_order_rmse[i][1][4]))
            f.write('\n')
    results_order_picp = sorted(results.items(), key=lambda x: x[1][3], reverse=True)
    for i in range(len(results_order_picp)):
        with open(path+'/results_picp.txt', 'a') as f:
            f.write(str(results_order_picp[i][0]) + ' RMSE: %f +- %f (stddev) +- %f (std error) PICP %f MPIW %f'
                    % (results_order_picp[i][1][0], results_order_picp[i][1][1], results_order_picp[i][1][2], results_order_picp[i][1][3],results_order_picp[i][1][4]))
            f.write('\n')