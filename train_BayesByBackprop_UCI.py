from __future__ import division, print_function
import time
import torch
import torch.utils.data
from torchvision import transforms, datasets
# torchvision包由流行的数据集（torchvision.datasets）、模型架构(torchvision.models)和用于计算机视觉的常见图像转换组成(torchvision.transforms)
import argparse
import matplotlib
from src.Bayes_By_Backprop.UCI.model import *
from src.Bayes_By_Backprop_Local_Reparametrization.UCI.model import *
from utils import readmts_uci_har,transform_labels
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Train Bayesian Neural Net on MNIST with Variational Inference')
parser.add_argument('--model', type=str, nargs='?', action='store', default='Gaussian_prior',
                    help='Model to run. Options are \'Gaussian_prior\', \'Laplace_prior\', \'GMM_prior\','
                         ' \'Local_Reparam\'. Default: \'Local_Reparam\'.')
parser.add_argument('--prior_sig', type=float, nargs='?', action='store', default=0.1,
                    help='Standard deviation of prior. Default: 0.1.')
parser.add_argument('--epochs', type=int, nargs='?', action='store', default=100,
                    help='How many epochs to train. Default: 200.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-4,
                    help='learning rate. Default: 1e-3.')
parser.add_argument('--n_samples', type=float, nargs='?', action='store', default=3,
                    help='How many MC samples to take when approximating the ELBO. Default: 3.')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='UCI_BBP_models',
                    help='Where to save learnt weights and train vectors. Default: \'BBP_models\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='UCI_BBP_results',
                    help='Where to save learnt training plots. Default: \'BBP_results\'.')
args = parser.parse_args()



# Where to save models weights
models_dir = args.models_dir
# Where to save plots and error, accuracy vectors
results_dir = args.results_dir

mkdir(models_dir)
mkdir(results_dir)
# ------------------------------------------------------------------------------------------------------
# train config
NTrainPointsMNIST = 9269
batch_size = 100
nb_epochs = args.epochs
log_interval = 1


# ------------------------------------------------------------------------------------------------------
# dataset

cprint('c', '\nData:')
file_name = './datasets/UCI_HAR_Dataset'
x_train, y_train, x_test, y_test = readmts_uci_har(file_name)
data = np.concatenate((x_train, x_test),axis=0)
label = np.concatenate((y_train, y_test),axis=0)
N = data.shape[0]
ind = int(N*0.9)
x_train = data[:ind]
y_train = label[:ind]
x_test = data[ind:]
y_test = label[ind:]
y_train, y_test = transform_labels(y_train, y_test)
NUM_CLASSES = 6

x_train = x_train.swapaxes(1,2)
x_test = x_test.swapaxes(1,2)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train)
print(x_train.size(), y_train.size())
trainset = torch.utils.data.TensorDataset(x_train, y_train)

x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test)
print(x_test.size(), y_test.size())
valset = torch.utils.data.TensorDataset(x_test, y_test)

# load data
# data augmentation
# transform_train = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
# ])

use_cuda = torch.cuda.is_available()

if use_cuda:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                              num_workers=3)
    #pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，
    #则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
    #当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                            num_workers=3)

else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                              num_workers=3)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                            num_workers=3)

## ---------------------------------------------------------------------------------------------------------------------
# net dims
cprint('c', '\nNetwork:')

lr = args.lr
nsamples = int(args.n_samples)  # How many samples to estimate ELBO with at each iteration
########################################################################################

if args.model == 'Local_Reparam':
    net = BBP_Bayes_Net_LR(lr=lr, channels_in=9, side_in=128, cuda=use_cuda, classes=6, batch_size=batch_size,
                     Nbatches=(NTrainPointsMNIST / batch_size), nhid=1200, prior_sig=args.prior_sig)
elif args.model == 'Laplace_prior':
    net = BBP_Bayes_Net(lr=lr, channels_in=9, side_in=128, cuda=use_cuda, classes=6, batch_size=batch_size,
                        Nbatches=(NTrainPointsMNIST / batch_size), nhid=1200,
                        prior_instance=laplace_prior(mu=0, b=args.prior_sig))
elif args.model == 'Gaussian_prior':
    net = BBP_Bayes_Net(lr=lr, channels_in=9, side_in=128, cuda=use_cuda, classes=6, batch_size=batch_size,
                        Nbatches=(NTrainPointsMNIST / batch_size), nhid=1200,
                        prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig))
elif args.model == 'GMM_prior':
    net = BBP_Bayes_Net(lr=lr, channels_in=9, side_in=128, cuda=use_cuda, classes=6, batch_size=batch_size,
                        Nbatches=(NTrainPointsMNIST / batch_size), nhid=1200,
                        prior_instance=spike_slab_2GMM(mu1=0, mu2=0, sigma1=args.prior_sig, sigma2=0.0005, pi=0.75))
else:
    print('Invalid model type')
    exit(1)

## ---------------------------------------------------------------------------------------------------------------------
# train
epoch = 0
cprint('c', '\nTrain:')

print('  init cost variables:')
kl_cost_train = np.zeros(nb_epochs)
pred_cost_train = np.zeros(nb_epochs)
err_train = np.zeros(nb_epochs)

cost_dev = np.zeros(nb_epochs)
err_dev = np.zeros(nb_epochs)
best_err = np.inf

nb_its_dev = 1

tic0 = time.time()
for i in range(epoch, nb_epochs):
    # We draw more samples on the first epoch in order to ensure convergence
    if i == 0:
        ELBO_samples = 10
    else:
        ELBO_samples = nsamples

    net.set_mode_train(True)
    tic = time.time()
    nb_samples = 0

    for x, y in trainloader:
        cost_dkl, cost_pred, err = net.fit(x, y, samples=ELBO_samples)

        err_train[i] += err
        kl_cost_train[i] += cost_dkl
        pred_cost_train[i] += cost_pred
        nb_samples += len(x)

    kl_cost_train[i] /= nb_samples  # Normalise by number of samples in order to get comparable number to the -log like
    pred_cost_train[i] /= nb_samples
    err_train[i] /= nb_samples

    toc = time.time()
    net.epoch = i
    # ---- print
    print("it %d/%d, Jtr_KL = %f, Jtr_pred = %f, err = %f, " % (
    i, nb_epochs, kl_cost_train[i], pred_cost_train[i], err_train[i]), end="")
    cprint('r', '   time: %f seconds\n' % (toc - tic))

    # ---- dev
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        nb_samples = 0
        for j, (x, y) in enumerate(valloader):
            cost, err, probs = net.eval(x, y)  # This takes the expected weights to save time, not proper inference

            cost_dev[i] += cost
            err_dev[i] += err
            nb_samples += len(x)

        cost_dev[i] /= nb_samples
        err_dev[i] /= nb_samples

        cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))

        if err_dev[i] < best_err:
            best_err = err_dev[i]
            cprint('b', 'best test error')
            net.save(models_dir + '/theta_best.dat')

toc0 = time.time()
runtime_per_it = (toc0 - tic0) / float(nb_epochs)
cprint('r', '   average time: %f seconds\n' % runtime_per_it)

net.save(models_dir + '/theta_last.dat')

## ---------------------------------------------------------------------------------------------------------------------
# results
cprint('c', '\nRESULTS:')
nb_parameters = net.get_nb_parameters()
best_cost_dev = np.min(cost_dev)
best_cost_train = np.min(pred_cost_train)
err_dev_min = err_dev[::nb_its_dev].min()

print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
print('  err_dev: %f' % (err_dev_min))
print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
print('  time_per_it: %fs\n' % (runtime_per_it))

## Save results for plots
# np.save('results/test_predictions.npy', test_predictions)
np.save(results_dir + '/KL_cost_train.npy', kl_cost_train)
np.save(results_dir + '/pred_cost_train.npy', pred_cost_train)
np.save(results_dir + '/cost_dev.npy', cost_dev)
np.save(results_dir + '/err_train.npy', err_train)
np.save(results_dir + '/err_dev.npy', err_dev)

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
lgd = plt.legend(['train error', 'test error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
ax = plt.gca()
plt.title('classification costs')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(textsize)
    item.set_weight('normal')
plt.savefig(results_dir + '/pred_cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(kl_cost_train, 'r')
ax1.set_ylabel('nats?')
plt.xlabel('epoch')
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='k', linestyle='--')
ax = plt.gca()
plt.title('DKL (per sample)')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(textsize)
    item.set_weight('normal')
plt.savefig(results_dir + '/KL_cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure(dpi=100)
fig2, ax2 = plt.subplots()
ax2.set_ylabel('% error')
ax2.semilogy(range(0, nb_epochs, nb_its_dev), 100 * err_dev[::nb_its_dev], 'b-')
ax2.semilogy(100 * err_train, 'r--')
plt.xlabel('epoch')
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='k', linestyle='--')
ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
lgd = plt.legend(['test error', 'train error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
ax = plt.gca()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(textsize)
    item.set_weight('normal')
plt.savefig(results_dir + '/err.png', bbox_extra_artists=(lgd,), box_inches='tight')
