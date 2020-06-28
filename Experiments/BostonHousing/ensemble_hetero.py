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
from src.Bootstrap_Ensemble.model import *
from Experiments.BostonHousing.utils import *
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # Load data
    X, Y = load_data()
    inputs = 13
    outputs = 1

    # Hyper-parameters
    subsamples = [0.8, 0.9]
    lrs = [1e-4, 1e-3]
    weight_decays = [1e-6, 1e-4]
    # n_nets = [10, 50, 100]
    n_nets = [10]
    momentums = [0, 0.9]
    
    NTrainPoints = 364
    batch_size = 100
    nb_epochs = 40
    log_interval = 1
    n_splits = 15

    # Paths
    base_dir = './results_hetero/ensemble_results'

    # Grid search
    results = {}
    for n_net in n_nets :
        for subsample in subsamples:
            for lr in lrs:
                for momentum in momentums:
                    for weight_decay in weight_decays:

                        Hps = 'N_net_' + str(n_net) + '_Subsample_' + str(subsample) + '_Lr_' + str(lr) + '_Momentum_' + str(momentum) + '_Weight_decay_' + str(weight_decay)
                        print('Grid search step: ' + Hps)

                        results_dir = base_dir + '/' + Hps
                        results_file = results_dir + '_results.txt'
                        mkdir(results_dir)

                        rmses = []
                        picps = []
                        mpiws = []

                        for split in range(int(n_splits)):
                            results_dir_split = results_dir + '/split_' + str(split)
                            mkdir(results_dir_split)

                            # get splited data\dataset
                            X_train, y_train, X_val, y_val, X_test, y_test, y_stds = get_data_splited(split, X, Y)
                            trainset, valset, testset = get_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

                            results_val = base_dir + '/results_val_split_' + str(split) + '.txt'
                            results_test = base_dir + '/results_test_split_' + str(split) + '.txt'

                            ###
                            output = np.zeros((X_test.shape[0], outputs * 2, n_net))
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
                                trainloader, valloader, testloader = get_dataloader_sample(trainset, valset, testset, use_cuda, batch_size, sampler, worker=False)

                                results_val_split = results_dir + '/results_val_split_' + str(split) + '.txt'
                                results_test_split = results_dir + '/results_test_split_' + str(split) + '.txt'

                                # net dims
                                cprint('c', '\nNetwork:')
                                net = Bootstrap_Net_BH_hetero(lr=lr, input_dim=inputs, output_dim=outputs, cuda=use_cuda,
                                                    batch_size=batch_size,
                                                    weight_decay=weight_decay, n_hid=50)

                                ## ---------------------------------------------------------------------------------------------------------------------
                                # train
                                epoch = 0
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
                                    print("it %d/%d, Jtr_pred = %f, rmse = %f, noise = %f" %
                                          (i, nb_epochs, pred_cost_train[i], rmse_train[i], net.model.log_noise.exp().cpu().data.numpy()), end="")
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

                                best_cost_dev = np.min(cost_dev)
                                best_cost_train = np.min(pred_cost_train)
                                rmse_dev_min = rmse_dev[::nb_its_dev].min()

                                print('  cost_test: %f ' % (cost_test))
                                print('  rmse_test: %f' % (rmse_test))
                                print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
                                print('  rmse_dev: %f' % (rmse_dev_min))
                                print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
                                print('  time_per_it: %fs\n' % (runtime_per_it))

                                # Storing validation result
                                store_results(results_val_split, ['Net_%d: rmse %f  \n' % (iii,rmse_dev_min * y_stds)])

                                # Storing testing result
                                store_results(results_test_split, ['Net_%d: rmse %f  \n' % (iii,rmse_test * y_stds)])


                            means = np.mean(output[:, :1, :], axis=2)
                            stds = np.std(output[:, :1, :], axis=2)
                            noises = np.mean(output[:, 1:, :] **2, axis=2)
                            # compute PICP MPIW
                            total_unc_1 = (noises ** 2 + stds ** 2) ** 0.5
                            total_unc_2 = (noises ** 2 + (2 * stds) ** 2) ** 0.5
                            total_unc_3 = (noises ** 2 + (3 * stds) ** 2) ** 0.5

                            y_L = means - total_unc_2
                            y_U = means + total_unc_2
                            u = np.maximum(0, np.sign(y_U - y_test))
                            l = np.maximum(0, np.sign(y_test - y_L))
                            PICP = np.mean(np.multiply(u, l))
                            MPIW = np.mean(y_U - y_L)

                            rmse_test_split = F.mse_loss(torch.tensor(means), torch.tensor(y_test), reduction='mean') **0.5
                            rmse_test_split = rmse_test_split.cpu().data.numpy()

                            # Storing testing results
                            store_results(results_test, [Hps + ' :: ', 'rmse %f PICP %f MPIW %f' % (rmse_test_split*y_stds, PICP, MPIW) + '\n'])

                            # storing testing results for this split
                            store_results(results_file, ['rmse %f PICP %f MPIW %f' % (rmse_test_split*y_stds, PICP, MPIW) + '\n'])

                            rmses.append(rmse_test_split*y_stds)
                            picps.append(PICP)
                            mpiws.append(MPIW)

                            shutil.rmtree(results_dir_split)
                            mkdir(results_dir_split)

                            ## plot figures
                            plot_uncertainty_noise(means, noises, [total_unc_1, total_unc_2, total_unc_3], y_test, results_dir_split)

                        rmses = np.array(rmses)
                        picps = np.array(picps)
                        mpiws = np.array(mpiws)

                        store_results(results_file,
                                      ['Overall: \n rmses %f +- %f (stddev) +- %f (std error) PICP %f MPIW %f\n' % (
                                          np.mean(rmses), np.std(rmses), np.std(rmses) / math.sqrt(n_splits),
                                          np.mean(picps), np.mean(mpiws))])

                        s = 'N_net: ' + str(n_net) + ' Subsample: ' + str(subsample) + \
                        ' Lr: ' + str(lr) + ' Momentum: ' + str(momentum) + ' Weight_decay: ' + str(weight_decay)

                        results[s] = [np.mean(rmses), np.std(rmses), np.std(rmses)/math.sqrt(n_splits), np.mean(picps), np.mean(mpiws)]

    # sort all the results
    store_all_results(results, base_dir)