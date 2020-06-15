from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib
from src.Bootstrap_Ensemble.model import *
import copy
from sklearn.datasets import load_boston
from pandas import Series,DataFrame

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_DATA_DIRECTORY_PATH = './data/'

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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Train Ensemble of MAP nets using bootstrapping')

parser.add_argument('--weight_decay', type=float, nargs='?', action='store', default=0,
                    help='Specify the precision of an isotropic Gaussian prior. Default: 0.')
parser.add_argument('--subsample', type=float, nargs='?', action='store', default=0.8,
                    help='Rate at which to subsample the dataset to train each net in the ensemble. Default: 0.8.')
parser.add_argument('--n_nets', type=int, nargs='?', action='store', default=100,
                    help='Number of nets in ensemble. Default: 100.')
parser.add_argument('--epochs', type=int, nargs='?', action='store', default=10,
                    help='How many epochs to train each net. Default: 10.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='learning rate. Default: 1e-3.')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='Ensemble_models',
                    help='Where to save learnt weights and train vectors. Default: \'Ensemble_models\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='Ensemble_results',
                    help='Where to save learnt training plots. Default: \'Ensemble_results\'.')
args = parser.parse_args()



# Where to save models weights
models_dir = args.models_dir
# Where to save plots and rmse, accuracy vectors
results_dir = args.results_dir

mkdir(models_dir)
mkdir(results_dir)
# ------------------------------------------------------------------------------------------------------
# train config
NTrainPoints = 364
batch_size = 128
nb_epochs = args.epochs
log_interval = 1

# ------------------------------------------------------------------------------------------------------
# dataset
boston = load_boston()
df = DataFrame(boston.data,columns=boston.feature_names)
X = boston.data  # 样本的特征值
Y = boston.target  # 样本的目标值

split = 0

results_dir = results_dir + '/split_' + str(split)
models_dir = models_dir + '/split_' + str(split)
mkdir(results_dir)
mkdir(models_dir)

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

## ---------------------------------------------------------------------------------------------------------------------
# net dims
cprint('c', '\nNetwork:')

lr = args.lr
weight_decay = args.weight_decay
########################################################################################
# This is The Bootstrapy part

Nruns = args.n_nets

weight_set_samples = []

p_subsample = args.subsample


############    Nruns:ensemble 数量
for iii in range(Nruns):
    keep_idx = []
    for idx in range(len(trainset)):

        if np.random.binomial(1, p_subsample, size=1) == 1:
            keep_idx.append(idx)

    keep_idx = np.array(keep_idx)

    from torch.utils.data.sampler import SubsetRandomSampler

    sampler = SubsetRandomSampler(keep_idx)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=3, sampler=sampler)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                num_workers=3)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                  num_workers=3, sampler=sampler)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                num_workers=3)

    ###############################################################
    net = Bootstrap_Net_BH(lr=lr, input_dim=inputs, cuda=use_cuda, output_dim=outputs, batch_size=batch_size,
                        weight_decay=weight_decay, n_hid=1200)

    epoch = 0

    ## ---------------------------------------------------------------------------------------------------------------------
    # train
    cprint('c', '\nTrain:')

    print('  init cost variables:')
    pred_cost_train = np.zeros(nb_epochs)
    rmse_train = np.zeros(nb_epochs)

    cost_dev = np.zeros(nb_epochs)
    rmse_dev = np.zeros(nb_epochs)
    # best_cost = np.inf
    best_rmse = np.inf

    nb_its_dev = 1

    tic0 = time.time()
    for i in range(epoch, nb_epochs):

        net.set_mode_train(True)

        tic = time.time()
        nb_samples = 0

        for x, y in trainloader:
            cost_pred, rmse = net.fit(x, y)

            rmse_train[i] += rmse
            pred_cost_train[i] += cost_pred
            nb_samples += len(x)

        pred_cost_train[i] /= nb_samples
        rmse_train[i] /= nb_samples

        toc = time.time()
        net.epoch = i
        # ---- print
        print("it %d/%d, Jtr_pred = %f, rmse = %f, " % (i, nb_epochs, pred_cost_train[i], rmse_train[i]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))

        # ---- dev
        if i % nb_its_dev == 0:
            net.set_mode_train(False)
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):
                cost, rmse  = net.eval(x, y)

                cost_dev[i] += cost
                rmse_dev[i] += rmse
                nb_samples += len(x)

            cost_dev[i] /= nb_samples
            rmse_dev[i] /= nb_samples

            cprint('g', '    Jdev = %f, rmse = %f\n' % (cost_dev[i], rmse_dev[i]))

            if rmse_dev[i] < best_rmse:
                best_rmse = rmse_dev[i]

    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    ## ---------------------------------------------------------------------------------------------------------------------
    # results
    cprint('c', '\nRESULTS:')
    nb_parameters = net.get_nb_parameters()
    best_cost_dev = np.min(cost_dev)
    best_cost_train = np.min(pred_cost_train)
    rmse_dev_min = rmse_dev[::nb_its_dev].min()

    print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
    print('  rmse_dev: %f' % (rmse_dev_min))
    print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
    print('  time_per_it: %fs\n' % (runtime_per_it))

    ########
    weight_set_samples.append(copy.deepcopy(net.model.state_dict()))


    ## ---------------------------------------------------------------------------------------------------------------------
    # fig cost vs its

    textsize = 15
    marker = 5

    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(pred_cost_train, 'r--')
    ax1.plot(range(0, nb_epochs, nb_its_dev), cost_dev[::nb_its_dev], 'b-')
    ax1.set_ylabel('Cross Entropy')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['train rmse', 'test rmse'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('classification costs')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/cost%d.png' % iii, bbox_extra_artists=(lgd,), bbox_inches='tight')

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
    lgd = plt.legend(['test rmse', 'train rmse'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/rmse%d.png' % iii, bbox_extra_artists=(lgd,), box_inches='tight')


save_object(weight_set_samples, models_dir+'/state_dicts.pkl')
