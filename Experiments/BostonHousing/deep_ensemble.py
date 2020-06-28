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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    # Load data
    X, Y = load_data()
    inputs = 13
    outputs = 1

    # Hyper-parameters
    n_nets = [5]
    subsamples = [0.9, 1]
    epsilons = [1e-2]
    alphas = [1]

    lrs = [1e-3]
    weight_decays = [1e-6, 1e-4]
    momentums = [0, 0.9]
    
    NTrainPoints = 364
    batch_size = 100
    nb_epochs = 400
    log_interval = 1
    n_splits = 3

    # Paths
    base_dir = './results/deep_ensemble_results'

    # Grid search
    results = {}
    for n_net in n_nets :
        for subsample in subsamples:
            for epsilon in epsilons:
                for alpha in alphas:
                    for lr in lrs:
                        for momentum in momentums:
                            for weight_decay in weight_decays:

                                Hps = 'N_net_' + str(n_net) + '_Subsample_' + str(subsample) + '_Epsilon_' + str(epsilon) \
                                      + '_Alpha_' + str(alpha) + '_Lr_' + str(lr) + '_Momentum_' + str(momentum) + '_Weight_decay_' + str(weight_decay)
                                print('Grid search step: ' + Hps)

                                results_dir = base_dir + '/' + Hps
                                results_file = results_dir + '_results.txt'
                                mkdir(results_dir)

                                rmses = []
                                picps = []
                                mpiws = []
                                nlls = []

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

                                    keep_idx = []
                                    for idx in range(len(trainset)):

                                        if np.random.binomial(1, subsample, size=1) == 1:
                                            keep_idx.append(idx)

                                    keep_idx = np.array(keep_idx)

                                    from torch.utils.data.sampler import SubsetRandomSampler

                                    sampler = SubsetRandomSampler(keep_idx)

                                    use_cuda = torch.cuda.is_available()
                                    trainloader, valloader, testloader = get_dataloader_sample(trainset, valset, testset,
                                                                                               use_cuda, batch_size, sampler, worker=False)

                                    results_val_split = results_dir + '/results_val_split_' + str(split) + '.txt'
                                    results_test_split = results_dir + '/results_test_split_' + str(split) + '.txt'

                                    # net dims
                                    cprint('c', '\nNetwork:')
                                    ensemble = [Deep_Ensemble_Net_BH(lr=lr, input_dim=inputs, output_dim=outputs, cuda=use_cuda,
                                                               batch_size=batch_size, weight_decay=weight_decay, n_hid=50,
                                                               momentum=momentum) for i in range(n_net)]

                                    ## ---------------------------------------------------------------------------------------------------------------------
                                    # train
                                    epoch = 0
                                    cprint('c', '\nTrain:')

                                    print('  init cost variables:')
                                    pred_cost_train = np.zeros((nb_epochs, n_net))
                                    cost_dev = np.zeros((nb_epochs, n_net))
                                    rmse_dev = np.zeros((nb_epochs, n_net))
                                    best_rmse = [np.inf for i in range(n_net)]
                                    best_cost = [np.inf for i in range(n_net)]

                                    nb_its_dev = 1
                                    tic0 = time.time()
                                    for i in range(epoch, nb_epochs):

                                        for iii in range(n_net):
                                            print('Net ' + str(iii))
                                            net = ensemble[iii]
                                            net.set_mode_train(True)

                                            tic = time.time()
                                            nb_samples = 0

                                            for x, y in trainloader:
                                                cost_pred = net.fit(x, y, epsilon, alpha)
                                                pred_cost_train[i, iii] += cost_pred
                                                nb_samples += len(x)

                                            pred_cost_train[i, iii] /= len(trainloader)
                                            toc = time.time()
                                            net.epoch = i

                                            # ---- print
                                            print("it %d/%d net %d, Jtr_pred = %f " %
                                                  (i, nb_epochs, iii, pred_cost_train[i, iii]), end="")
                                            cprint('r', '   time: %f seconds\n' % (toc - tic))

                                            # ---- dev
                                            if i % nb_its_dev == 0:
                                                net.set_mode_train(False)
                                                nb_samples = 0
                                                for j, (x, y) in enumerate(valloader):
                                                    cost, mse, _ = net.eval(x, y)
                                                    cost_dev[i, iii] += cost
                                                    rmse_dev[i, iii] += mse
                                                    nb_samples += len(x)

                                                cost_dev[i, iii] /= len(valloader)
                                                rmse_dev[i, iii] = (rmse_dev[i, iii] / nb_samples)**0.5

                                                cprint('g', ' net %d,  Jdev = %f, err = %f\n' % (iii, cost_dev[i, iii], rmse_dev[i, iii]))

                                                # if rmse_dev[i] < best_rmse:
                                                #     best_rmse = rmse_dev[i]
                                                if cost_dev[i, iii] < best_cost[iii]:
                                                    best_cost[iii] = cost_dev[i, iii]
                                                    cprint('b', 'net %d, best val loss(nll)' %(iii))
                                                    net.save(results_dir_split + '/theta_best_val_' + str(iii) + '.dat')

                                    toc0 = time.time()
                                    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
                                    cprint('r', '   average time: %f seconds\n' % runtime_per_it)
                                    ## ---------------------------------------------------------------------------------------------------------------------
                                    # results
                                    for iii in range(n_net):
                                        net = ensemble[iii]
                                        net.load(results_dir_split + '/theta_best_val_' + str(iii) + '.dat')
                                        cprint('c', '\n Net %d RESULTS:' %(iii))
                                        nb_parameters = net.get_nb_parameters()

                                        net.set_mode_train(False)
                                        nb_samples = 0
                                        cost_test = 0
                                        rmse_test = 0

                                        start = 0
                                        for j, (x, y) in enumerate(testloader):
                                            end = start + len(x)
                                            cost, mse, out = net.eval(x, y)
                                            output[start:end, :, iii] = out.cpu().numpy()
                                            start = end
                                            cost_test += cost
                                            rmse_test += mse
                                            nb_samples += len(x)

                                        cost_test /= len(testloader)
                                        rmse_test = (rmse_test / nb_samples) ** 0.5

                                        cost_test = cost_test.cpu().data.numpy()
                                        rmse_test = rmse_test.cpu().data.numpy()

                                        best_cost_dev = np.min(cost_dev[:, iii])
                                        best_cost_train = np.min(pred_cost_train[:, iii])
                                        rmse_dev_min = rmse_dev[:, iii][::nb_its_dev].min()

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

                                        # Plotting
                                        mkdir(results_dir_split + '/net_' + str(iii))
                                        print(results_dir_split + '/net_' + str(iii))
                                        plot_pred_cost(pred_cost_train[:, iii], nb_epochs, nb_its_dev, cost_dev[:, iii],results_dir_split + '/net_' + str(iii))
                                        plot_rmse(nb_epochs, nb_its_dev, None, rmse_dev[:, iii], results_dir_split + '/net_' + str(iii))

                                    output_mu = output[:,:1,:]
                                    output_sig = output[:,1:,:]
                                    sig_pos = np.log(1 + np.exp(output_sig)) + 1e-06
                                    means = np.mean(output_mu, axis=2)
                                    vars = np.mean(sig_pos + np.square(output_mu), axis=2) - np.square(means)
                                    stds = np.sqrt(vars)

                                    # compute PICP MPIW
                                    y_L = means - 2 * stds
                                    y_U = means + 2 * stds
                                    u = np.maximum(0, np.sign(y_U - y_test))
                                    l = np.maximum(0, np.sign(y_test - y_L))
                                    PICP = np.mean(np.multiply(u, l))
                                    MPIW = np.mean(y_U - y_L)

                                    rmse_test_split = F.mse_loss(torch.tensor(means), torch.tensor(y_test), reduction='mean') **0.5
                                    nll_test_split = gaussian_nll(torch.tensor(means), torch.tensor(vars), torch.tensor(y_test))
                                    rmse_test_split = rmse_test_split.cpu().data.numpy()
                                    nll_test_split = nll_test_split.cpu().data.numpy()

                                    # Storing testing results
                                    store_results(results_test, [Hps + ' :: ', 'nll %f rmse %f PICP %f MPIW %f' % (nll_test_split, rmse_test_split*y_stds, PICP, MPIW) + '\n'])

                                    # storing testing results for this split
                                    store_results(results_file, ['nll %f rmse %f PICP %f MPIW %f' % (nll_test_split, rmse_test_split*y_stds, PICP, MPIW) + '\n'])

                                    rmses.append(rmse_test_split*y_stds)
                                    nlls.append(nll_test_split)
                                    picps.append(PICP)
                                    mpiws.append(MPIW)

                                    mkdir(results_dir_split)

                                    ## plot figures
                                    plot_uncertainty(means, stds, y_test, results_dir_split)

                                rmses = np.array(rmses)
                                nlls = np.array(nlls)
                                picps = np.array(picps)
                                mpiws = np.array(mpiws)

                                store_results(results_file,
                                              ['Overall: \n rmses %f +- %f (stddev) +- %f (std error) nll %f PICP %f MPIW %f\n' % (
                                                  np.mean(rmses), np.std(rmses), np.std(rmses) / math.sqrt(n_splits),
                                                  np.mean(nlls), np.mean(picps), np.mean(mpiws))])

                                s = 'N_net: ' + str(n_net) + ' Subsample: ' + str(subsample) + '_Epsilon_' + str(epsilon) \
                                      + '_Alpha_' + str(alpha) + ' Lr: ' + str(lr) + ' Momentum: ' + str(momentum) + ' Weight_decay: ' + str(weight_decay)

                                results[s] = [np.mean(rmses), np.std(rmses), np.std(rmses)/math.sqrt(n_splits), np.mean(nlls), np.mean(picps), np.mean(mpiws)]

    # sort all the results
    results_order_rmse = sorted(results.items(), key=lambda x: x[1][0], reverse=False)
    for i in range(len(results_order_rmse)):
        with open(base_dir + '/results_rmse.txt', 'a') as f:
            f.write(str(results_order_rmse[i][0]) + ' RMSE: %f +- %f (stddev) +- %f (std error) nll %f PICP %f MPIW %f'
                    % (results_order_rmse[i][1][0], results_order_rmse[i][1][1], results_order_rmse[i][1][2],
                       results_order_rmse[i][1][3], results_order_rmse[i][1][4], results_order_rmse[i][1][5]))
            f.write('\n')
    results_order_picp = sorted(results.items(), key=lambda x: x[1][4], reverse=True)
    for i in range(len(results_order_picp)):
        with open(base_dir + '/results_picp.txt', 'a') as f:
            f.write(str(results_order_picp[i][0]) + ' RMSE: %f +- %f (stddev) +- %f (std error) nll %f PICP %f MPIW %f'
                    % (results_order_picp[i][1][0], results_order_picp[i][1][1], results_order_picp[i][1][2],
                       results_order_picp[i][1][3], results_order_picp[i][1][4], results_order_rmse[i][1][5]))
            f.write('\n')
    results_order_nll = sorted(results.items(), key=lambda x: x[1][4], reverse=True)
    for i in range(len(results_order_nll)):
        with open(base_dir + '/results_nll.txt', 'a') as f:
            f.write(str(results_order_nll[i][0]) + ' RMSE: %f +- %f (stddev) +- %f (std error) nll %f PICP %f MPIW %f'
                    % (results_order_nll[i][1][0], results_order_nll[i][1][1], results_order_nll[i][1][2],
                       results_order_nll[i][1][3], results_order_nll[i][1][4], results_order_rmse[i][1][5]))
            f.write('\n')