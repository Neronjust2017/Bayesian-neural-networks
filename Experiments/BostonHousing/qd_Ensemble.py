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
from src.Quality_deiven_PI_Ensemble.model import *
from src.Quality_deiven_PI_Ensemble.utils import *
from Experiments.BostonHousing.utils import *
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    # Load data
    X, Y = load_data()
    inputs = 13
    outputs = 1

    # type_in = '~' + 'boston'  # data type to use - drunk_bow_tie x_cubed_gap ~boston ~concrete
    # loss_type = 'gauss_like'  # loss type to train on - qd_soft gauss_like(=mve) mse (mse=simple point prediction)
    # n_samples = 10000  # if generating data, how many points to generate
    # h_size = [50]  # number of hidden units in network: [50]=layer_1 of 50, [8,4]=layer_1 of 8, layer_2 of 4
    # alpha = 0.05  # data points captured = (1 - alpha)
    # n_epoch = 100  # number epochs to train for
    # optim = 'adam'  # opitimiser - SGD adam
    # l_rate = 0.03  # learning rate of optimiser
    # decay_rate = 0.9  # learning rate decay
    # soften = 160.  # hyper param for QD_soft
    # lambda_in = 15.  # hyper param for QD_soft
    # sigma_in = 0.2  #  initialise std dev of NN weights
    # is_run_test = True  # if averaging over lots of runs - turns off some prints and graphs
    # n_ensemble = 5  # number of individual NNs in ensemble
    # n_bootstraps = 1  # how many boostrap resamples to perform
    # n_runs = 20 if is_run_test else 1
    # is_batch = True  # train in batches?
    # n_batch = 100  # batch size
    # lube_perc = 90.  # if model uncertainty method = perc - 50 to 100
    # perc_or_norm = 'norm'  # model uncertainty method - perc norm (paper uses norm)
    # is_early_stop = False  # stop training early (didn't use in paper)
    # is_bootstrap = False if n_bootstraps == 1 else True
    # train_prop = 0.9  # % of data to use as training, 0.8 for hyperparam selection
    #
    # out_biases = [3., -3.]  # chose biases for output layer (for gauss_like is overwritten to 0,1)
    # activation = 'relu'  # NN activation fns - tanh relu

    # Hyper-parameters
    subsamples = [0.8]
    lrs = [0.03]
    alphas = [0.05, 0.10]     # data points captured = (1 - alpha)
    loss_types = ['qd_soft', 'qd_hard']    # loss type to train on - qd_soft mve mse (mse=simple point prediction)
    censor_Rs = [False]
    type_ins = ["pred_intervals"]
    softens = [160.]    # hyper param for QD_soft
    lambda_ins = [5., 10.]
    bias_rands = [False]
    out_biases = [[3.,-3.]] # chose biases for output layer (for mve is overwritten to 0,1)
    weight_decays = [1e-6]
    n_nets = [10]
    momentums = [0.99]

    lube_perc = 90  # if model uncertainty method = perc - 50 to 100
    perc_or_norm = 'norm'   # model uncertainty method - perc norm (paper uses norm)

    batch_size = 100
    nb_epochs = 100
    log_interval = 1
    n_splits = 15

    # plotting options
    save_graphs = True
    show_graphs = True
    is_y_rescale = False
    is_y_sort = False
    var_plot = 0  # lets us plot against different variables, use 0 for univariate
    is_err_bars = True
    is_norm_plot = True

    # Paths
    base_dir = './results/qd_ensemble_results_new'
    
    # Grid search
    results = {}
    for n_net in n_nets :
        for subsample in subsamples:
            for alpha in alphas:
                for loss_type in loss_types:
                    for censor_R in censor_Rs:
                        for type_in in type_ins:
                            for soften in softens:
                                for lambda_in in lambda_ins:
                                    for bias_rand in bias_rands:
                                        for out_biase in out_biases:
                                             for lr in lrs:
                                                for momentum in momentums:
                                                    for weight_decay in weight_decays:

                                                        # pre calcs
                                                        if alpha == 0.05:
                                                            n_std_devs = 1.96
                                                        elif alpha == 0.10:
                                                            n_std_devs = 1.645
                                                        elif alpha == 0.01:
                                                            n_std_devs = 2.575
                                                        else:
                                                            raise Exception('ERROR unusual alpha')

                                                        Hps = 'N_net_' + str(n_net) + '_Subsample_' + str(subsample) + '_Alpha_' + str(alpha) +\
                                                              '_Lr_' + str(lr) + '_Momentum_' + str(momentum) + '_Weight_decay_' + str(weight_decay) \
                                                              + '_loss_type_' + str(loss_type) + '_Type_ins_' + type_in + '_Soften_' + str(soften) \
                                                              + '_lambda_in_' + str(lambda_in)

                                                        print('Grid search step: ' + Hps)

                                                        results_dir = base_dir + '/' + Hps
                                                        results_file = results_dir + '_results.txt'
                                                        mkdir(results_dir)

                                                        results_splits = []

                                                        for split in range(int(n_splits)):
                                                            results_dir_split = results_dir + '/split_' + str(split)
                                                            mkdir(results_dir_split)

                                                            # get splited data\dataset
                                                            X_train, y_train, X_val, y_val, X_test, y_test, y_stds = get_data_splited(split, X, Y)
                                                            trainset, valset, testset = get_dataset(X_train,y_train, X_val,y_val, X_test,y_test)

                                                            results_val = base_dir + '/results_val_split_' + str(split) + '.txt'
                                                            results_test = base_dir + '/results_test_split_' + str(split) + '.txt'

                                                            ###
                                                            y_pred_all = np.zeros((n_net, X_test.shape[0], outputs * 2))
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
                                                                trainloader, valloader, testloader = get_dataloader_sample(trainset, valset, testset, use_cuda, batch_size, sampler)

                                                                results_val_split = results_dir + '/results_val_split_' + str(split) + '.txt'
                                                                results_test_split = results_dir + '/results_test_split_' + str(split) + '.txt'

                                                                # net dims
                                                                cprint('c', '\nNetwork:')
                                                                net = QD_net_BH( lr=lr, input_dim=inputs, cuda=use_cuda, output_dim=outputs, batch_size=128,
                                                                                 type_in=type_in, alpha=alpha,loss_type=loss_type, censor_R=censor_R,
                                                                                 soften=soften, lambda_in=lambda_in,  bias_rand=bias_rand, out_biases=out_biases,
                                                                                 weight_decay=weight_decay, n_hid=50, momentum=momentum)
                                                                ## ---------------------------------------------------------------------------------------------------------------------
                                                                # train
                                                                epoch = 0
                                                                cprint('c', '\nTrain:')

                                                                print('  init cost variables:')
                                                                pred_cost_train = []
                                                                PICP_train = []
                                                                MPIW_train = []

                                                                cost_dev = []
                                                                PICP_dev = []
                                                                MPIW_dev = []

                                                                best_val_loss = np.inf

                                                                nb_its_dev = 1

                                                                tic0 = time.time()

                                                                early_stop = 0
                                                                for i in range(epoch, nb_epochs):

                                                                    net.set_mode_train(True)

                                                                    tic = time.time()
                                                                    nb_samples = 0

                                                                    pred_cost_train_i = 0
                                                                    picp_train_i = 0
                                                                    mpiw_train_i = 0

                                                                    for x, y in trainloader:
                                                                        cost_pred, picp, mpiw = net.fit(x, y)
                                                                        pred_cost_train_i += cost_pred
                                                                        picp_train_i += picp
                                                                        mpiw_train_i += mpiw
                                                                        nb_samples += len(x)

                                                                    pred_cost_train_i /= len(trainloader)
                                                                    picp_train_i /= len(trainloader)
                                                                    mpiw_train_i /= len(trainloader)

                                                                    pred_cost_train.append(pred_cost_train_i)
                                                                    PICP_train.append(picp_train_i)
                                                                    MPIW_train.append(mpiw_train_i)

                                                                    toc = time.time()
                                                                    net.epoch = i

                                                                    # ---- print
                                                                    print("it %d/%d, loss_train = %f, PICP_train = %f, MPIW_train = %f " % (
                                                                    i, nb_epochs, pred_cost_train_i, picp_train_i, mpiw_train_i), end="")
                                                                    cprint('r', '   time: %f seconds\n' % (toc - tic))

                                                                    # ---- dev
                                                                    if i % nb_its_dev == 0:
                                                                        net.set_mode_train(False)
                                                                        cost_dev_i = 0
                                                                        picp_dev_i = 0
                                                                        mpiw_dev_i = 0

                                                                        for j, (x, y) in enumerate(valloader):
                                                                            cost, picp, mpiw, _ = net.eval(x, y)

                                                                            cost_dev_i += cost
                                                                            picp_dev_i += picp
                                                                            mpiw_dev_i += mpiw
                                                                            nb_samples += len(x)

                                                                        cost_dev_i /= len(valloader)
                                                                        picp_dev_i /= len(valloader)
                                                                        mpiw_dev_i /= len(valloader)

                                                                        cost_dev.append(cost_dev_i)
                                                                        PICP_dev.append(picp_dev_i)
                                                                        MPIW_dev.append(mpiw_dev_i)

                                                                        cprint('g', ' loss_val = %f, PICP_val = %f, MPIW_val = %f\n' % (cost_dev_i, picp_dev_i, mpiw_dev_i))

                                                                        if cost_dev_i < best_val_loss:
                                                                            best_val_loss = cost_dev_i
                                                                            early_stop = 0
                                                                            cprint('b', 'best_val_loss')
                                                                            net.save(results_dir_split + '/theta_best_val_' + str(iii) + '.dat')
                                                                        else:
                                                                            early_stop += 1

                                                                        if early_stop > 20 and epoch > nb_epochs/2:
                                                                            break

                                                                toc0 = time.time()
                                                                runtime_per_it = (toc0 - tic0) / float(nb_epochs)
                                                                cprint('r', '   average time: %f seconds\n' % runtime_per_it)
                                                                ## ---------------------------------------------------------------------------------------------------------------------
                                                                # results
                                                                net.load(results_dir_split + '/theta_best_val_' + str(iii) + '.dat')

                                                                cprint('c', '\nRESULTS:')
                                                                nb_parameters = net.get_nb_parameters()

                                                                net.set_mode_train(False)
                                                                cost_test = 0
                                                                picp_test = 0
                                                                mpiw_test = 0

                                                                start = 0
                                                                for j, (x, y) in enumerate(testloader):
                                                                    end = start + len(x)
                                                                    cost, picp, mpiw, out = net.eval(x, y)
                                                                    y_pred_all[iii, start:end, :] = out.cpu().numpy()
                                                                    start = end
                                                                    cost_test += cost
                                                                    picp_test += picp
                                                                    mpiw_test += mpiw

                                                                cost_test /= len(testloader)
                                                                picp_test /= len(testloader)
                                                                mpiw_test /= len(testloader)

                                                                best_cost_dev = np.min(np.array(cost_dev))
                                                                best_cost_train = np.min(np.array(pred_cost_train))
                                                                picp_dev_min = np.array(PICP_dev)[::nb_its_dev].min()
                                                                mpiw_dev_min = np.array(MPIW_dev)[::nb_its_dev].min()

                                                                print('  cost_test: %f ' % (cost_test))
                                                                print('  picp_test: %f' % (picp_test))
                                                                print('  mpiw_test: %f' % (mpiw_test))

                                                                print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
                                                                print('  picp_dev: %f' % (picp_dev_min))
                                                                print('  mpiw_dev: %f' % (mpiw_dev_min))

                                                                print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
                                                                print('  time_per_it: %fs\n' % (runtime_per_it))

                                                                # Storing validation results
                                                                store_results(results_val_split, ['Net_%d: PICP %f MPIW %f\n' % (iii, picp_dev_min, mpiw_dev_min)])

                                                                # Storing testing results
                                                                store_results(results_test_split, ['Net_%d: PICP %f MPIW %f  \n' % (iii, picp_test, mpiw_test)])

                                                            if loss_type == 'qd_soft':
                                                                y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, \
                                                                y_pred_L = pi_to_gauss(y_pred_all, lube_perc,perc_or_norm, n_std_devs)

                                                            elif loss_type == 'gauss_like':  # work out bounds given mu sigma
                                                                y_pred_gauss_mid_all = y_pred_all[:, :, 0]
                                                                # occasionally may get -ves for std dev so need to do max
                                                                y_pred_gauss_dev_all = np.sqrt(np.maximum(
                                                                    np.log(1. + np.exp(y_pred_all[:, :, 1])),
                                                                    10e-6))
                                                                y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, \
                                                                y_pred_L = gauss_to_pi(y_pred_gauss_mid_all,
                                                                                       y_pred_gauss_dev_all,
                                                                                       n_std_devs)

                                                            elif loss_type == 'mse':  # as for gauss_like but we don't know std dev so guess
                                                                y_pred_gauss_mid_all = y_pred_all[:, :, 0]
                                                                y_pred_gauss_dev_all = np.zeros_like(
                                                                    y_pred_gauss_mid_all) + 0.01
                                                                y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, \
                                                                y_pred_L = gauss_to_pi(y_pred_gauss_mid_all,
                                                                                       y_pred_gauss_dev_all,
                                                                                       n_std_devs)

                                                            # work out metrics
                                                            y_U_cap = y_pred_U > y_test.reshape(-1)
                                                            y_L_cap = y_pred_L < y_test.reshape(-1)
                                                            y_all_cap = y_U_cap * y_L_cap
                                                            PICP = np.sum(y_all_cap) / y_L_cap.shape[0]
                                                            MPIW = np.mean(y_pred_U - y_pred_L)
                                                            y_pred_mid = np.mean((y_pred_U, y_pred_L), axis=0)
                                                            MSE = np.mean(np.square(
                                                                y_stds * (y_pred_mid - y_test[:, 0])))
                                                            RMSE = np.sqrt(MSE)
                                                            CWC = np_QD_loss(y_test, y_pred_L, y_pred_U, alpha,
                                                                             soften, lambda_in)
                                                            neg_log_like = gauss_neg_log_like(y_test,
                                                                                              y_pred_gauss_mid,
                                                                                              y_pred_gauss_dev,
                                                                                              y_stds)
                                                            residuals = residuals = y_pred_mid - y_test[:, 0]
                                                            shapiro_W, shapiro_p = stats.shapiro(residuals[:])
                                                            results_splits.append((MSE, RMSE, PICP, MPIW, CWC,
                                                                                 neg_log_like, shapiro_W,
                                                                                 shapiro_p))

                                                            # concatenate for graphs
                                                            title = 'PICP=' + str(round(PICP, 3)) \
                                                                    + ', MPIW=' + str(round(MPIW, 3)) \
                                                                    + ', qd_loss=' + str(round(CWC, 3)) \
                                                                    + ', NLL=' + str(round(neg_log_like, 3)) \
                                                                    + ', alpha=' + str(alpha) \
                                                                    + ', loss=' + loss_type \
                                                                    + ', data=' + type_in + ',' \
                                                                    + ', ensemb=' + str(n_nets) \
                                                                    + ', RMSE=' + str(round(RMSE, 3)) \
                                                                    + ', soft=' + str(soften) \
                                                                    + ', lambda=' + str(lambda_in)

                                                            # visualise
                                                            if show_graphs:
                                                                # error bars
                                                                if is_err_bars:

                                                                    save_path = results_dir + '/split_' + str(
                                                                        split) + '_err_bars.png'

                                                                    plot_err_bars(X_test, y_test, y_pred_U,
                                                                                  y_pred_L,
                                                                                  is_y_sort, is_y_rescale,
                                                                                  y_stds, save_graphs,save_path,
                                                                                  title, var_plot)

                                                                # normal dist stuff
                                                                if is_norm_plot:
                                                                    title = 'shapiro_W=' + str(
                                                                        round(shapiro_W, 3)) + \
                                                                            ', data=' + type_in + ', loss=' + loss_type + \
                                                                            ', n_test=' + str(y_test.shape[0])
                                                                    fig, (ax1, ax2) = plt.subplots(2)
                                                                    ax1.set_xlabel(
                                                                        'y_pred - y_test')  # histogram
                                                                    ax1.hist(residuals, bins=30)
                                                                    ax1.set_title(title, fontsize=10)
                                                                    stats.probplot(residuals[:],
                                                                                   plot=ax2)  # QQ plot
                                                                    ax2.set_title('')
                                                                    fig.show()
                                                                    plt.savefig(results_dir + '/split_' + str(
                                                                        split) + '_norm_plot.png',
                                                                                bbox_inches='tight')

                                                            store_results(results_test, [Hps + ' :: ', 'mse %f rmse %f PICP %f MPIW %f CWC %f nll %f shapiro_W %f shapiro_p %f '
                                                                             % (MSE, RMSE, PICP, MPIW, CWC, neg_log_like, shapiro_W, shapiro_p) + '\n'])

                                                            store_results(results_file, ['mse %f rmse %f PICP %f MPIW %f CWC %f nll %f shapiro_W %f shapiro_p %f '
                                                                    % (MSE, RMSE, PICP, MPIW, CWC, neg_log_like,shapiro_W, shapiro_p) + '\n'])

                                                            shutil.rmtree(results_dir_split)

                                                            c = ['#1f77b4', '#ff7f0e']
                                                            ind = np.arange(0, len(y_test))
                                                            plt.figure()
                                                            fig, ax1 = plt.subplots()
                                                            plt.scatter(ind, y_test, color='black', alpha=0.5)
                                                            ax1.plot(ind, y_pred_mid, 'r')
                                                            plt.fill_between(ind, y_pred_L, y_pred_U,
                                                                             alpha=0.25)

                                                            ax1.set_ylabel('prediction')
                                                            plt.xlabel('test points')
                                                            plt.grid(b=True, which='major', color='k', linestyle='-')
                                                            plt.grid(b=True, which='minor', color='k', linestyle='--')
                                                            ax = plt.gca()
                                                            plt.title('Uncertainty')

                                                            plt.savefig(results_dir + '/split_' + str(split) + '_uncertainty.png',
                                                                        bbox_inches='tight')

                                                        results_splits.append((MSE, RMSE, PICP, MPIW, CWC,
                                                                               neg_log_like, shapiro_W,
                                                                               shapiro_p))
                                                        results_splits = np.array(results_splits)

                                                        means = np.mean(results_splits, axis=0)
                                                        stds = np.std(results_splits[:, 1])
                                                        std_errors = stds / math.sqrt(n_splits)

                                                        store_results(results_file, ['Overall: \n rmse %f +- %f (stddev) +- %f (std error) PICP %f MPIW %f CWC %f nll %f shapiro_W %f shapiro_p %f '
                                                                % ( means[1], stds, std_errors, means[2], means[3], means[4], means[5], means[6], means[7]) + '\n'])

                                                        s = 'N_net: ' + str(n_net) + ' Subsample:' + str(
                                                            subsample) + ' Alpha: ' + str(alpha) + \
                                                              ' Lr: ' + str(lr) + ' Momentum: ' + str(
                                                            momentum) + ' Weight_decay: ' + str(weight_decay) \
                                                              + ' loss_type: ' + str(
                                                            loss_type) + ' Type_ins: ' + type_in + ' Soften: ' + str(
                                                            soften) \
                                                              + ' lambda_in: ' + str(lambda_in)

                                                        results[s] = [ means[1], stds, std_errors, means[2], means[3], means[4], means[5], means[6], means[7]]

    results_order_rmse = sorted(results.items(), key=lambda x: x[1][0], reverse=False)
    for i in range(len(results_order_rmse)):
        with open(base_dir + '/results_rmse.txt', 'a') as f:
            f.write(str(results_order_rmse[i][0]) + ' RMSE: %f +- %f (stddev) +- %f (std error) PICP %f MPIW %f CWC %f nll %f shapiro_W %f shapiro_p %f '
                    % (results_order_rmse[i][1][0], results_order_rmse[i][1][1], results_order_rmse[i][1][2],
                       results_order_rmse[i][1][3], results_order_rmse[i][1][4], results_order_rmse[i][1][5],
                       results_order_rmse[i][1][6], results_order_rmse[i][1][7], results_order_rmse[i][1][8]))
            f.write('\n')
    results_order_picp = sorted(results.items(), key=lambda x: x[1][3], reverse=True)
    for i in range(len(results_order_picp)):
        with open(base_dir + '/results_picp.txt', 'a') as f:
            f.write(str(results_order_picp[i][0]) + ' RMSE: %f +- %f (stddev) +- %f (std error) PICP %f MPIW %f CWC %f nll %f shapiro_W %f shapiro_p %f '
                    % (results_order_picp[i][1][0], results_order_picp[i][1][1], results_order_picp[i][1][2],
                       results_order_picp[i][1][3], results_order_picp[i][1][4], results_order_picp[i][1][5],
                       results_order_picp[i][1][6], results_order_picp[i][1][7], results_order_picp[i][1][8]))
            f.write('\n')