# %%

import GPy
import time
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.sgd import SGD

from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
# from google.colab import files
# % config
# InlineBackend.figure_format = 'svg'

# %%

torch.cuda.device(0)
torch.cuda.get_device_name(torch.cuda.current_device())


# %%

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:

        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


# %%
def log_gaussian_loss_1(output, target, sigma, no_dim):
    exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma)

    return -(log_coeff + exponent).sum()

def log_gaussian_loss(output, target):
    # exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    exponent = -0.5 * (target - output) ** 2
    # log_coeff = -no_dim * torch.log(sigma)

    return -exponent.sum()


def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)

    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()

    return (varpost_lik * (varpost_loglik - prior_loglik)).sum()


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def loglik(self, weights):
        exponent = -0.5 * (weights - self.mu) ** 2 / self.sigma ** 2
        log_coeff = -0.5 * (np.log(2 * np.pi) + 2 * np.log(self.sigma))

        return (exponent + log_coeff).sum()


# %%

class MC_Dropout_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob):
        super(MC_Dropout_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.biases = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))

    def forward(self, x):
        dropout_mask = torch.bernoulli((1 - self.dropout_prob) * torch.ones(self.weights.shape)).cuda()

        return torch.mm(x, self.weights * dropout_mask) + self.biases

# %%

class MC_Dropout_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units, init_log_noise):
        super(MC_Dropout_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer1 = nn.Linear(input_dim, no_units)
        self.layer2 = nn.Linear(no_units, no_units)
        self.layer3 = nn.Linear(no_units, no_units)
        self.layer4 = nn.Linear(no_units, no_units)
        self.layer5 = nn.Linear(no_units, output_dim)

        # activation to be used between hidden layers
        # self.activation = nn.ReLU(inplace=True)
        self.activation = nn.Tanh()
        self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        x = self.layer1(x)
        x = self.activation(x)

        x = F.dropout(x, p=0.1, training=True)

        x = self.layer2(x)
        x = self.activation(x)

        x = F.dropout(x, p=0.1, training=True)

        x = self.layer3(x)
        x = self.activation(x)

        x = F.dropout(x, p=0.1, training=True)

        x = self.layer4(x)
        x = self.activation(x)

        x = F.dropout(x, p=0.1, training=True)

        x = self.layer5(x)

        return x


# %%

class MC_Dropout_Wrapper:
    def __init__(self, input_dim, output_dim, no_units, learn_rate, batch_size, no_batches, weight_decay,
                 init_log_noise):
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches

        self.network = MC_Dropout_Model(input_dim=input_dim, output_dim=output_dim,
                                        no_units=no_units, init_log_noise=0)
        self.network.cuda()

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learn_rate, weight_decay=weight_decay)
        self.loss_func = log_gaussian_loss_1

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=True)

        # reset gradient and total loss
        self.optimizer.zero_grad()

        output = self.network(x)
        # print(output)
        loss = self.loss_func(output, y, torch.exp(self.network.log_noise), 1) / len(x)
        # print(y)

        loss.backward()
        self.optimizer.step()

        return loss


import h5py
f = h5py.File('./data/train.h5','r')
x_train = f['data'].value
y_train = f['label'].value

f = h5py.File('./data/test.h5','r')
x_test = f['data'].value
y_test = f['label'].value

plt.figure(figsize=(50, 5))
plt.style.use('default')
plt.plot(x_train, y_train, color='black',linewidth=1)
plt.plot(x_test, y_test, color='red',linewidth=1)
plt.show()

num_epochs, batch_size, nb_train = 1000, len(x_train), len(x_train)

net = MC_Dropout_Wrapper(input_dim=1, output_dim=1, no_units=1024, learn_rate=1e-2,
                         batch_size=batch_size, no_batches=1, init_log_noise=0, weight_decay=1e-6)

for i in range(num_epochs):

    # net.network.train(mode=True)
    loss = net.fit(x_train, y_train)

    if i % 200 == 0:
        print('Epoch: %4d, Train loss = %7.3f' % \
              (i, loss.cpu().data.numpy()))

# %%

samples_train = []
samples_test = []
for i in range(1000):
    preds = net.network.forward(torch.from_numpy(x_train).float().cuda()).cpu().data.numpy()
    samples_train.append(preds)
    preds = net.network.forward(torch.from_numpy(x_test).float().cuda()).cpu().data.numpy()
    samples_test.append(preds)

samples_train = np.array(samples_train)
means_train = (samples_train.mean(axis=0)).reshape(-1)
epistemic_train = (samples_train.var(axis=0) ** 0.5).reshape(-1)

samples_test = np.array(samples_test)
means_test = (samples_test.mean(axis=0)).reshape(-1)
epistemic_test = (samples_test.var(axis=0) ** 0.5).reshape(-1)

c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

x_test = x_test.reshape([x_test.shape[0],])
y_test = y_test.reshape([y_test.shape[0],])
x_train = x_train.reshape([x_train.shape[0],])
y_train = y_train.reshape([y_train.shape[0],])

plt.figure(figsize=(50, 5))
plt.style.use('default')
plt.plot(x_train, y_train, color='black',linewidth=1)

plt.fill_between(x_train, means_train - epistemic_train, means_train + epistemic_train, color=c[0], alpha=0.1,
                 label='Epistemic')
plt.plot(x_train, means_train, color='red', linewidth=1)


plt.plot(x_test, y_test, color='black',linewidth=1)

plt.fill_between(x_test, means_test - epistemic_test, means_test + epistemic_test, color=c[0], alpha=0.4,
                 label='Epistemic')
plt.plot(x_test, means_test, color='red', linewidth=1)


plt.xlabel('$x$', fontsize=30)
plt.title('MC dropout', fontsize=40)
plt.tick_params(labelsize=30)
# plt.xticks(np.arange(0, 400, 20))
# plt.yticks(np.arange(-2, 2, 0.5))
plt.gca().set_yticklabels([])
plt.gca().yaxis.grid(alpha=0.3)
plt.gca().xaxis.grid(alpha=0.3)
plt.savefig('mc_dropout.png', bbox_inches='tight')

# files.download("mc_dropout.pdf")

plt.show()

# %%

# samples = []
# noises = []
# for i in range(1000):
#     preds = net.network.forward(torch.linspace(-5, 5, 200).cuda()).cpu().data.numpy()
#     samples.append(preds)
#
# samples = np.array(samples)
# means = (samples.mean(axis=0)).reshape(-1)
#
# c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#
# plt.figure(figsize=(6, 5))
# plt.style.use('default')
# plt.scatter(x_train, y_train, s=10, marker='x', color='black', alpha=0.5)
#
# plt.plot(np.linspace(-5, 5, 200), means, color='black', linewidth=1)
# plt.xlim([-5, 5])
# plt.ylim([-5, 7])
# plt.xlabel('$x$', fontsize=30)
# plt.title('MC dropout', fontsize=40)
# plt.tick_params(labelsize=30)
# plt.xticks(np.arange(-4, 5, 2))
# plt.yticks(np.arange(-4, 7, 2))
# plt.gca().set_yticklabels([])
# plt.gca().yaxis.grid(alpha=0.3)
# plt.gca().xaxis.grid(alpha=0.3)
# plt.savefig('mc_dropout.png', bbox_inches='tight')
#
# # files.download("mc_dropout.pdf")
#
# plt.show()
#
# # %%
