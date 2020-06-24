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
from src.Bayes_By_Backprop.model import *
from src.Bayes_By_Backprop_Local_Reparametrization.model import *
from Experiments.BostonHousing.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    # Load data
    X, Y = load_data()
    inputs = 13
    outputs = 1

    # Hyper-parameters
    priors = ['GMM_prior','Laplace_prior', 'Gaussian_prior']
    prior_sigs = [10, 1, 0.1, 0.05]
    lrs = [1e-4, 1e-3]
    momentums = [0, 0.9]
    n_samples = [10, 100]

    NTrainPoints = 364
    batch_size = 100
    nb_epochs = 40
    log_interval = 1
    n_splits = 15

    # Paths
    base_dir = './bbb_results'

    # Grid search
    results = {}
    for prior in priors :
        for prior_sig in prior_sigs:
            for lr in lrs:
                for momentum in momentums:
                    for n_sample in n_samples:

                        Hps = 'Prior_' + str(prior) + '_Prior_sigs_' + str(prior_sig) + '_Lr_' + str(lr) + '_Momentum_' + str(momentum) + '_N_sample_' + str(n_sample)
                        print('Grid search step: ' + Hps)

                        results_dir = base_dir + '/' + Hps
                        results_file = results_dir + '_results.txt'
                        mkdir(results_dir)

                        rmses = []
                        picps = []
                        mpiws = []

                        global prior_instance
                        if prior == 'Gaussian_prior':
                            prior_instance = isotropic_gauss_prior(mu=0, sigma=prior_sig)
                        elif prior == 'Laplace_prior':
                            prior_instance = laplace_prior(mu=0, b=prior_sig)
                        elif prior == 'GMM_prior':
                            prior_instance = spike_slab_2GMM(mu1=0, mu2=0, sigma1=prior_sig, sigma2=0.0005, pi=0.75)
                        else:
                            print('Invalid prior type')
                            exit(1)

                        for split in range(int(n_splits)):

                            results_dir_split = results_dir + '/split_' + str(split)
                            mkdir(results_dir_split)

                            # get splited data\dataset\dataloder
                            X_train, y_train, X_val, y_val, X_test, y_test, y_stds = get_data_splited(split, X, Y)
                            trainset, valset, testset = get_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

                            use_cuda = torch.cuda.is_available()

                            trainloader, valloader, testloader = get_dataloader(trainset, valset, testset, use_cuda,batch_size)

                            results_val = base_dir + '/results_val_split_' + str(split) + '.txt'
                            results_test = base_dir + '/results_test_split_' + str(split) + '.txt'

                            # net dims
                            cprint('c', '\nNetwork:')
                            net = BBP_Bayes_Net_BH(lr=lr, input_dim=inputs, output_dim=outputs, cuda=use_cuda,
                                                batch_size=batch_size,
                                                Nbatches=(NTrainPoints / batch_size), nhid=50,
                                                prior_instance=prior_instance)

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
                                # We draw more samples on the first epoch in order to ensure convergence
                                if i == 0:
                                    ELBO_samples = 10
                                else:
                                    ELBO_samples = n_sample

                                net.set_mode_train(True)
                                tic = time.time()
                                nb_samples = 0


                                for x, y in trainloader:
                                    cost_dkl, cost_pred, mse = net.fit(x, y, samples=ELBO_samples)
                                    kl_cost_train[i] += cost_dkl
                                    pred_cost_train[i] += cost_pred
                                    rmse_train[i] += mse
                                    nb_samples += len(x)

                                kl_cost_train[i] /= nb_samples  # Normalise by number of samples in order to get comparable number to the -log like
                                pred_cost_train[i] /= nb_samples
                                rmse_train[i] = (rmse_train[i] / nb_samples)**0.5

                                toc = time.time()
                                net.epoch = i
                                # ---- print
                                print("it %d/%d, Jtr_KL = %f, Jtr_pred = %f, rmse = %f, " % (
                                    i, nb_epochs, kl_cost_train[i], pred_cost_train[i], rmse_train[i]), end="")
                                cprint('r', '   time: %f seconds\n' % (toc - tic))

                                # ---- dev
                                if i % nb_its_dev == 0:
                                    net.set_mode_train(False)
                                    nb_samples = 0
                                    T = 1000

                                    for j, (x, y) in enumerate(valloader):
                                        cost, mse, _, _ = net.eval(x, y, samples=T)  # This takes the expected weights to save time, not proper inference
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
                            T = 1000

                            means = np.zeros((X_test.shape[0], outputs))
                            stds = np.zeros((X_test.shape[0], outputs))

                            start = 0
                            for j, (x, y) in enumerate(testloader):
                                end = len(x) + start
                                cost, mse, mean, std = net.eval(x, y, samples=T)  # This takes the expected weights to save time, not proper inference
                                if use_cuda:
                                    mean = mean.cpu()
                                    std = std.cpu()
                                means[start:end,:] = mean
                                stds[start:end,:] = std
                                start = end

                                cost_test += cost
                                rmse_test += mse
                                nb_samples += len(x)

                            # compute PICP MPIW
                            y_L = means - 2 * stds
                            y_U = means + 2 * stds
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
                            # np.save('results/test_predictions.npy', test_predictions)
                            np.save(results_dir_split + '/KL_cost_train.npy', kl_cost_train)
                            np.save(results_dir_split + '/pred_cost_train.npy', pred_cost_train)
                            np.save(results_dir_split + '/cost_dev.npy', cost_dev)
                            np.save(results_dir_split + '/rmse_train.npy', rmse_train)
                            np.save(results_dir_split + '/rmse_dev.npy', rmse_dev)

                            np.save(results_dir_split + '/means.npy', means)
                            np.save(results_dir_split + '/stds.npy', stds)

                            # Storing validation results
                            store_results(results_val, [Hps + ' :: ', 'rmse %f ' % (rmse_dev_min * y_stds) + '\n'])

                            # Storing testing results
                            store_results(results_test, [Hps + ' :: ', 'rmse %f PICP %f MPIW %f' % (
                            rmse_test * y_stds, PICP, MPIW) + '\n'])

                            # storing testing results for this split
                            store_results(results_file,
                                          ['rmse %f PICP %f MPIW %f' % (rmse_test * y_stds, PICP, MPIW) + '\n'])

                            ## ---------------------------------------------------------------------------------------------------------------------
                            ## plot figures
                            plot_pred_cost(pred_cost_train, nb_epochs, nb_its_dev, cost_dev, results_dir_split)
                            plot_kl_cost(kl_cost_train, results_dir_split)
                            plot_rmse(nb_epochs, nb_its_dev, rmse_train, rmse_dev, results_dir_split)
                            plot_uncertainty(means, stds, y_test, results_dir_split)

                        rmses = np.array(rmses)
                        picps = np.array(picps)
                        mpiws = np.array(mpiws)

                        store_results(results_file,['Overall: \n rmses %f +- %f (stddev) +- %f (std error) PICP %f MPIW %f\n' % (
                                          np.mean(rmses), np.std(rmses), np.std(rmses) / math.sqrt(n_splits),
                                          np.mean(picps), np.mean(mpiws))])

                        s = 'Prior: ' + str(prior) + ' Prior_sigs: ' + str(prior_sig) + \
                        ' Lr: ' + str(lr) + ' Momentum: ' + str(momentum) + ' N_sample: ' + str(n_sample)

                        results[s] = [np.mean(rmses), np.std(rmses)/int(n_splits), np.mean(picps), np.mean(mpiws)]

    # sort all the results
    store_all_results(results, base_dir)




