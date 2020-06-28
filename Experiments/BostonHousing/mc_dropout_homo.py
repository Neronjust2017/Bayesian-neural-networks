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
from src.utils import mkdir
from src.MC_dropout.model import *
from Experiments.BostonHousing.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == '__main__':
    # Load data
    X, Y = load_data()
    inputs = 13
    outputs = 1

    # Hyper-parameters
    # pdrops = [0.005, 0.01, 0.05, 0.1]
    # taus = [0.1, 0.15, 0.2]
    # lengthscales = [1e-2, 1e-1, 1, 10]
    # lrs = [1e-3, 1e-4]
    # momentums = [0.9]
    # Ts = [1000]
    pdrops = [0.2, 0.1]
    taus = [0.1, 0.15]
    lengthscales = [1e-1, 1]
    lrs = [1e-3]
    momentums = [0.9]
    Ts = [1000]

    NTrainPoints = 364
    batch_size = 128
    nb_epochs = 40
    log_interval = 1
    n_splits = 15
    
    # Paths
    base_dir = './results_homo/mc_dropout_results'

    # Grid search
    results = {}
    for pdrop in pdrops :
        for tau in taus:
            for lengthscale in lengthscales:
                for T in Ts:
                    for lr in lrs:
                        for momentum in momentums:

                            Hps = 'Pdrop_' + str(pdrop) + '_Tau_' + str(tau) + '_Lengthscale_' + str(lengthscale) \
                                  + '_Lr_' + str(lr) + '_Momentum_' + str(momentum) + '_T_' + str(T)
                            print('Grid search step:' + Hps )

                            results_dir = base_dir + '/' + Hps
                            results_file = results_dir + '_results.txt'
                            mkdir(results_dir)

                            rmses = []
                            picps = []
                            mpiws = []

                            for split in range(int(n_splits)):

                                results_dir_split = results_dir + '/split_' + str(split)
                                mkdir(results_dir_split)

                                # get splited data\dataset\dataloder
                                X_train, y_train, X_val, y_val, X_test, y_test, y_stds = get_data_splited(split, X, Y)
                                trainset, valset, testset = get_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

                                use_cuda = torch.cuda.is_available()

                                trainloader, valloader, testloader = get_dataloader(trainset, valset, testset, use_cuda,
                                                                                    batch_size)

                                results_val = base_dir + '/results_val_split_' + str(split) + '.txt'
                                results_test = base_dir + '/results_test_split_' + str(split) + '.txt'

                                # net dims
                                N = X_train.shape[0]
                                reg = lengthscale ** 2 * (1 - pdrop) / (2. * N * tau)

                                cprint('c', '\nNetwork:')
                                net = MC_drop_net_BH_homo(lr=lr, input_dim=inputs, output_dim=outputs, cuda=use_cuda,
                                                    batch_size=batch_size, weight_decay=reg,n_hid=50, momentum=momentum, pdrop=pdrop)

                                # ---- train
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

                                    toc = time.time()
                                    net.epoch = i
                                    # ---- print
                                    print("it %d/%d, Jtr_pred = %f, rmse = %f, noise = %f" % (
                                        i, nb_epochs, pred_cost_train[i], rmse_train[i], net.model.log_noise.exp().cpu().data.numpy()), end="")
                                    cprint('r', '   time: %f seconds\n' % (toc - tic))

                                    # ---- dev
                                    if i % nb_its_dev == 0:
                                        net.set_mode_train(False)
                                        nb_samples = 0

                                        for j, (x, y) in enumerate(valloader):
                                            cost, mse, _, _ = net.eval(x, y, samples=T)
                                            cost_dev[i] += cost
                                            rmse_dev[i] += mse
                                            nb_samples += len(x)

                                        cost_dev[i] /= nb_samples
                                        rmse_dev[i] = (rmse_dev[i] / nb_samples)**0.5

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

                                means = np.zeros((X_test.shape[0], outputs))
                                stds = np.zeros((X_test.shape[0], outputs))

                                # ---- test
                                start = 0
                                for j, (x, y) in enumerate(testloader):
                                    end = len(x) + start
                                    cost, mse, mean, std = net.eval(x, y, samples=T)
                                    if use_cuda:
                                        mean = mean.cpu()
                                        std = std.cpu()
                                    means[start:end, :] = mean
                                    stds[start:end, :] = std
                                    start = end

                                    cost_test += cost
                                    rmse_test += mse
                                    nb_samples += len(x)

                                # compute PICP MPIW
                                noise = net.model.log_noise.exp().cpu().data.numpy()
                                total_unc_1 = (noise ** 2 + stds ** 2) ** 0.5
                                total_unc_2 = (noise ** 2 + (2 * stds) ** 2) ** 0.5
                                total_unc_3 = (noise ** 2 + (3 * stds) ** 2) ** 0.5

                                y_L = means - total_unc_2
                                y_U = means + total_unc_2
                                u = np.maximum(0, np.sign(y_U - y_test))
                                l = np.maximum(0, np.sign(y_test - y_L))
                                PICP = np.mean(np.multiply(u, l))
                                MPIW = np.mean(y_U - y_L)

                                cost_test /= nb_samples
                                rmse_test = (rmse_test / nb_samples)**0.5

                                cost_test = cost_test.cpu().data.numpy()
                                rmse_test = rmse_test.cpu().data.numpy()

                                rmses.append(rmse_test*y_stds)
                                picps.append(PICP)
                                mpiws.append(MPIW)

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
                                np.save(results_dir_split + '/means.npy', means)
                                np.save(results_dir_split + '/stds.npy', stds)

                                # Storing validation results
                                store_results(results_val, [Hps + ' :: ', 'rmse %f ' % (rmse_dev_min * y_stds) + '\n'])

                                # Storing testing results
                                store_results(results_test, [Hps + ' :: ', 'rmse %f PICP %f MPIW %f' % (rmse_test * y_stds, PICP, MPIW) + '\n'])

                                # storing testing results for this split
                                store_results(results_file,
                                              ['rmse %f PICP %f MPIW %f' % (rmse_test * y_stds, PICP, MPIW) + '\n'])

                                ## ---------------------------------------------------------------------------------------------------------------------
                                ## plot figures
                                plot_pred_cost(pred_cost_train, nb_epochs, nb_its_dev, cost_dev, results_dir_split)
                                plot_rmse(nb_epochs, nb_its_dev, rmse_train, rmse_dev, results_dir_split)
                                plot_uncertainty_noise(means, noise, [total_unc_1, total_unc_2, total_unc_3], y_test, results_dir_split)

                            rmses = np.array(rmses)
                            picps = np.array(picps)
                            mpiws = np.array(mpiws)

                            store_results(results_file,['Overall: \n rmses %f +- %f (stddev) +- %f (std error) PICP %f MPIW %f\n' % (
                                              np.mean(rmses), np.std(rmses), np.std(rmses) / math.sqrt(n_splits),
                                              np.mean(picps), np.mean(mpiws))])

                            s = 'Pdrop: ' + str(pdrop) + ' Tau: ' + str(tau) + \
                            ' Lengthscale: ' + str(lengthscale) + ' Lr: ' + str(lr) + ' Momentum: ' + str(momentum) + ' T: ' + str(T)

                            results[s] = [np.mean(rmses), np.std(rmses), np.std(rmses)/math.sqrt(n_splits), np.mean(picps), np.mean(mpiws)]

    # sort all the results
    store_all_results(results, base_dir)
