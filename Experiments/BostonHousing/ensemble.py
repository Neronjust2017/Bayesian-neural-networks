import json

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.datasets import load_boston
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath
for i in range(2):
    rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

import numpy as np
import torch
import time
import math
from pandas import Series,DataFrame
import argparse
import matplotlib
from src.utils import mkdir
from src.Bootstrap_Ensemble.model import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def _get_index_train_test_path(split_num, train = True):
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

if __name__ == '__main__':
    # Data
    boston = load_boston()
    df = DataFrame(boston.data,columns=boston.feature_names)
    X = boston.data  # 样本的特征值
    Y = boston.target  # 样本的目标值

    _DATA_DIRECTORY_PATH = './data/'

    # subsamples = [0.8, 0.9]
    # lrs = [1e-4, 1e-3, 1e-2]
    # weight_decays = [0]
    # n_nets = [2, 200]
    # momentums = [0, 0.9]
    subsamples = [0.8]
    lrs = [1e-4]
    weight_decays = [1e-6]
    n_nets = [10, 50]
    momentums = [0]

    batch_size = 100
    nb_epochs = 40
    log_interval = 1

    results = {}
    for n_net in n_nets :
        for subsample in subsamples:
            for lr in lrs:
                for momentum in momentums:
                    for weight_decay in weight_decays:
                        print('Grid search step: N_net: ' + str(n_net) + ' Subsample: ' + str(subsample) + \
                              ' Lr: ' + str(lr) + ' Momentum: ' + str(momentum) + ' Weight_decay: ' + str(weight_decay))

                        results_dir = './ensemble_results/N_net_' + str(n_net) + '_Subsample_' + str(subsample) + \
                                      '_Lr_' + str(lr) + '_Momentum_' + str(momentum) + '_Weight_decay_' + str(weight_decay)

                        results_file = results_dir + '_results.txt'
                        mkdir(results_dir)

                        rmses = []

                        n_splits = 15

                        for split in range(int(n_splits)):
                            results_dir_split = results_dir + '/split_' + str(split)
                            mkdir(results_dir_split)

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

                            inputs = 13
                            outputs = 1

                            results_val = './bbb_results/results_val_split_' + str(split) + '.txt'
                            results_test = './bbb_results/results_test_split_' + str(split) + '.txt'

                            ###
                            output = np.zeros((x_test.shape[0], outputs, n_net))
                            ###
                            for iii in range(n_net):
                                print('Net ' + str(iii))
                                keep_idx = []
                                for idx in range(len(trainset)):

                                    if np.random.binomial(1, subsample, size=1) == 1:
                                        keep_idx.append(idx)

                                keep_idx = np.array(keep_idx)

                                from torch.utils.data.sampler import SubsetRandomSampler

                                sampler = SubsetRandomSampler(keep_idx)

                                use_cuda = torch.cuda.is_available()

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

                                results_val_split = results_dir + '/results_val_split_' + str(split) + '.txt'
                                results_test_split = results_dir + '/results_test_split_' + str(split) + '.txt'

                                ###############################################################
                                net = Bootstrap_Net_BH(lr=lr, input_dim=inputs, output_dim=outputs, cuda=use_cuda,
                                                    batch_size=batch_size,
                                                    weight_decay=weight_decay, n_hid=50)

                                epoch = 0

                                ## ---------------------------------------------------------------------------------------------------------------------
                                # train
                                cprint('c', '\nTrain:')
                                print('  init cost variables:')
                                kl_cost_train = np.zeros(nb_epochs)
                                pred_cost_train = np.zeros(nb_epochs)
                                rmse_train = np.zeros(nb_epochs)

                                cost_dev = np.zeros(nb_epochs)
                                rmse_dev = np.zeros(nb_epochs)
                                best_rmse = np.inf

                                nb_its_dev = 1

                                tic0 = time.time()
                                for i in range(epoch, nb_epochs):

                                    net.set_mode_train(True)

                                    tic = time.time()
                                    nb_samples = 0

                                    for x, y in trainloader:
                                        cost_pred = net.fit(x, y)

                                        rmse_train[i] += cost_pred
                                        pred_cost_train[i] += cost_pred
                                        nb_samples += len(x)

                                    pred_cost_train[i] /= nb_samples
                                    rmse_train[i] = (rmse_train[i] / nb_samples)**0.5

                                    toc = time.time()
                                    net.epoch = i

                                    # ---- print
                                    print("it %d/%d, Jtr_pred = %f, rmse = %f, " % (
                                    i, nb_epochs, pred_cost_train[i], rmse_train[i]), end="")
                                    cprint('r', '   time: %f seconds\n' % (toc - tic))

                                    # ---- dev
                                    if i % nb_its_dev == 0:
                                        net.set_mode_train(False)
                                        nb_samples = 0
                                        for j, (x, y) in enumerate(valloader):
                                            cost, out = net.eval(x, y)

                                            cost_dev[i] += cost
                                            rmse_dev[i] += cost
                                            nb_samples += len(x)

                                        cost_dev[i] /= nb_samples
                                        rmse_dev[i] = (rmse_dev[i] / nb_samples)**0.5

                                        cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev[i], rmse_dev[i]))

                                        if rmse_dev[i] < best_rmse:
                                            best_rmse = rmse_dev[i]
                                            cprint('b', 'best val rmse')
                                            net.save(results_dir_split + '/theta_best_val_' + str(iii) + '.dat')

                                toc0 = time.time()
                                runtime_per_it = (toc0 - tic0) / float(nb_epochs)
                                cprint('r', '   average time: %f seconds\n' % runtime_per_it)
                                ## ---------------------------------------------------------------------------------------------------------------------
                                # results
                                net.load(results_dir_split + '/theta_best_val_' + str(iii) + '.dat')
                                ###
                                #  删除
                                ###
                                cprint('c', '\nRESULTS:')
                                nb_parameters = net.get_nb_parameters()

                                net.set_mode_train(False)

                                nb_samples = 0
                                cost_test = 0
                                rmse_test = 0

                                start = 0
                                for j, (x, y) in enumerate(testloader):
                                    end = start + len(x)
                                    cost, out = net.eval(x, y)
                                    output[start:end, :, iii] = out.cpu().numpy()
                                    start = end
                                    cost_test += cost
                                    rmse_test += cost
                                    nb_samples += len(x)


                                cost_test /= nb_samples
                                rmse_test = (rmse_test / nb_samples) ** 0.5

                                cost_test = cost_test.cpu().data.numpy()
                                rmse_test = rmse_test.cpu().data.numpy()

                                # ###################
                                # cost_test *= y_stds
                                # rmse_test *= y_stds
                                # ###################

                                best_cost_dev = np.min(cost_dev)
                                best_cost_train = np.min(pred_cost_train)
                                rmse_dev_min = rmse_dev[::nb_its_dev].min()

                                print('  cost_test: %f ' % (cost_test))
                                print('  rmse_test: %f' % (rmse_test))

                                print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
                                print('  rmse_dev: %f' % (rmse_dev_min))
                                print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
                                print('  time_per_it: %fs\n' % (runtime_per_it))

                                # Storing validation results
                                with open(results_val_split, "a") as myfile:
                                    myfile.write('Net_%d: rmse %f  \n' % (iii,rmse_dev_min * y_stds))

                                # Storing testing results
                                with open(results_test_split, "a") as myfile:
                                    myfile.write('Net_%d: rmse %f  \n' % (iii,rmse_test * y_stds))

                            means = np.mean(output, axis=2)
                            stds = np.std(output, axis=2)

                            rmse_test_split = F.mse_loss(torch.tensor(means), y_test, reduction='mean')

                            with open(results_test, "a") as myfile:
                                myfile.write('N_net: ' + str(n_net) + ' Subsample: ' + str(subsample) + \
                              ' Lr: ' + str(lr) + ' Momentum: ' + str(momentum) + ' Weight_decay: ' + str(weight_decay) + ' :: ')
                                myfile.write('rmse %f ' % (rmse_test_split * y_stds) + '\n')

                            with open(results_file, "a") as myfile:
                                myfile.write('rmse %f  \n' % (rmse_test_split * y_stds))

                            rmses.append(rmse_test_split*y_stds)

                            shutil.rmtree(results_dir_split)

                            means = means.reshape((means.shape[0],))
                            stds = means.reshape((stds.shape[0],))

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
                            ax = plt.gca()
                            plt.title('Uncertainty')

                            plt.savefig(results_dir + '/split_' + str(split) + '_uncertainty.png',
                                        bbox_inches='tight')

                        rmses = np.array(rmses)
                        with open(results_file, "a") as myfile:
                            myfile.write('Overall: \n rmses %f +- %f (stddev)  \n' % (
                                np.mean(rmses), np.std(rmses) / int(n_splits)))

                        s = 'N_net: ' + str(n_net) + ' Subsample: ' + str(subsample) + \
                        ' Lr: ' + str(lr) + ' Momentum: ' + str(momentum) + ' Weight_decay: ' + str(weight_decay)

                        results[s] = [np.mean(rmses), np.std(rmses) / int(n_splits)]

    results_order = sorted(results.items(), key=lambda x: x[1][0], reverse=False)
    file = open('./ensemble_results/results.txt', 'w')
    file.writelines(json.dumps(results_order))
    file.close()