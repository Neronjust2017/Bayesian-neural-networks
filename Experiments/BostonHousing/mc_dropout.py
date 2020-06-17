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
from src.MC_dropout.model import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    pdrops = [0.5, 0.1]
    taus = [1.0]
    lengthscales = [1e-2]
    lrs = [1e-4]
    momentums = [0.9]
    Ts = [3]

    NTrainPoints = 364
    batch_size = 100
    nb_epochs = 2
    log_interval = 1

    for pdrop in pdrops :
        for tau in taus:
            for lengthscale in lengthscales:
                for T in Ts:
                    for lr in lrs:
                        for momentum in momentums:
                    
                            print('Grid search step: Pdrop: ' + str(pdrop) + ' Tau: ' + str(tau) + \
                            ' Lr: ' + str(lr) + ' Momentum: ' + str(momentum) + ' T: ' + str(T))

                            results_dir = './mc_dropout_results/Pdrop_' + str(pdrop) + '_Tau_' + str(tau) + \
                            '_Lr_' + str(lr) + '_Momentum_' + str(momentum) + '_T_' + str(T)

                            results_file = results_dir + '_results.txt'
                            mkdir(results_dir)

                            rmses = []
                            # rmse_stds = 0

                            n_splits = 5

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

                                use_cuda = torch.cuda.is_available()

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

                                results_val = './mc_dropout_results/results_val_split_' + str(split) + '.txt'
                                results_test = './mc_dropout_results/results_test_split_' + str(split) + '.txt'

                                # net dims
                                N = X_train.shape[0]

                                reg = lengthscale ** 2 * (1 - pdrop) / (2. * N * tau)

                                cprint('c', '\nNetwork:')
                                net = MC_drop_net_BH(lr=lr, input_dim=inputs, output_dim=outputs, cuda=use_cuda,
                                                    batch_size=batch_size, weight_decay=reg,n_hid=50, momentum=momentum, pdrop=pdrop)

                                # train
                                epoch = 0
                                cprint('c', '\nTrain:')

                                print('  init cost variables:')

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
                                        pred_cost_train[i] += cost_pred
                                        rmse_train[i] += cost_pred
                                        nb_samples += len(x)

                                    pred_cost_train[i] /= nb_samples
                                    rmse_train[i] = (rmse_train[i] / nb_samples)**0.5

                                    # ###################
                                    # pred_cost_train[i] *= y_stds
                                    # rmse_train[i] *= y_stds
                                    # ###################

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
                                        # rmses_dev = np.zeros((X_train.shape[0], outputs, T))
                                        # start = 0
                                        for j, (x, y) in enumerate(valloader):
                                            # end = len(x) + start
                                            cost, mse = net.eval(x, y, samples=T)
                                            # start = end
                                            cost_dev[i] += cost
                                            rmse_dev[i] += mse
                                            nb_samples += len(x)

                                        cost_dev[i] /= nb_samples
                                        rmse_dev[i] = (rmse_dev[i] / nb_samples)**0.5

                                        # ###################
                                        # cost_dev[i] *= y_stds
                                        # rmse_dev[i] *= y_stds
                                        # ###################

                                        # rmse_std_dev = np.std(np.mean(rmses_dev))

                                        cprint('g', '    Jdev = %f, rmse = %f\n' % (cost_dev[i], rmse_dev[i]))

                                        if rmse_dev[i] < best_rmse:
                                            best_rmse = rmse_dev[i]
                                            cprint('b', 'best val rmse')
                                            net.save(results_dir_split + '/theta_best_val.dat')

                                toc0 = time.time()
                                runtime_per_it = (toc0 - tic0) / float(nb_epochs)
                                cprint('r', '   average time: %f seconds\n' % runtime_per_it)
                                ## ---------------------------------------------------------------------------------------------------------------------
                                # results
                                net.load(results_dir_split + '/theta_best_val.dat')
                                cprint('c', '\nRESULTS:')
                                nb_parameters = net.get_nb_parameters()

                                net.set_mode_train(False)
                                nb_samples = 0
                                cost_test = 0
                                rmse_test = 0

                                # rmses_test = np.zeros((X_train.shape[0], outputs, T))
                                # start = 0
                                for j, (x, y) in enumerate(testloader):
                                    # end = len(x) + start
                                    cost, mse = net.eval(x, y, samples=T)
                                    cost_test += cost
                                    rmse_test += mse
                                    nb_samples += len(x)

                                cost_test /= nb_samples
                                rmse_test = (rmse_test / nb_samples)**0.5
                                # rmse_std_test = np.std(np.mean(rmses_test))

                                cost_test = cost_test.cpu().data.numpy()
                                rmse_test = rmse_test.cpu().data.numpy()

                                # ###################
                                # cost_test *= y_stds
                                # rmse_test *= y_stds
                                # ###################

                                rmses.append(rmse_test*y_stds)
                                # rmse_stds += rmse_std_test

                                best_cost_dev = np.min(cost_dev)
                                best_cost_train = np.min(pred_cost_train)
                                rmse_dev_min = rmse_dev[::nb_its_dev].min()

                                print('  cost_test: %f ' % (cost_test))
                                print('  rmse_test: %f' % (rmse_test))

                                print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
                                print('  rmse_dev: %f' % (rmse_dev_min))
                                print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
                                print('  time_per_it: %fs\n' % (runtime_per_it))

                                ## Save results for plots
                                np.save(results_dir_split + '/pred_cost_train.npy', pred_cost_train)
                                np.save(results_dir_split + '/cost_dev.npy', cost_dev)
                                np.save(results_dir_split + '/rmse_train.npy', rmse_train)
                                np.save(results_dir_split + '/rmse_dev.npy', rmse_dev)

                                # Storing validation results
                                with open(results_val, "a") as myfile:
                                    myfile.write('Pdrop_' + str(pdrop) + '_Tau_' + str(tau) + \
                            '_Lr_' + str(lr) + '_Momentum_' + str(momentum) + '_T_' + str(T) + ' :: ')
                                    myfile.write('rmse %f ' % (rmse_dev_min*y_stds) + '\n')

                                # Storing testing results
                                with open(results_test, "a") as myfile:
                                    myfile.write('Pdrop_' + str(pdrop) + '_Tau_' + str(tau) + \
                                '_Lr_' + str(lr) + '_Momentum_' + str(momentum) + '_T_' + str(T) + ' :: ')
                                    myfile.write('rmse %f ' % (rmse_test*y_stds) + '\n')

                                with open(results_file, "a") as myfile:
                                    myfile.write('rmse %f  \n' % (rmse_test*y_stds))

                                ## ---------------------------------------------------------------------------------------------------------------------
                                # fig cost vs its
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

                                plt.figure(dpi=100)
                                fig2, ax2 = plt.subplots()
                                ax2.set_ylabel('% rmse')
                                ax2.semilogy(range(0, nb_epochs, nb_its_dev), 100 * rmse_dev[::nb_its_dev], 'b-')
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

                            rmses = np.array(rmses)
                            with open(results_file, "a") as myfile:
                                myfile.write('Overall: \n rmses %f +- %f (stddev)  \n' % (
                                    np.mean(rmses), np.std(rmses)/int(n_splits)))


