from src.priors import *
from src.base_net import *

import torch.nn.functional as F
import torch.nn as nn


def MC_dropout(act_vec, p=0.5, mask=True):
    # return F.dropout(act_vec, p=p, training=mask, inplace=True)
    return F.dropout(act_vec, p=p, training=mask, inplace=False)

class Linear_1L(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid, pdrop):
        super(Linear_1L, self).__init__()

        self.pdrop = pdrop

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, output_dim)

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)

    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        # -----------------
        x = self.fc1(x)
        x = self.act(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        # -----------------
        y = self.fc2(x)

        return y

class Linear_1L_homo(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid, pdrop, init_log_noise=0):
        super(Linear_1L_homo, self).__init__()

        self.pdrop = pdrop

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, output_dim)

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)
        self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))

    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        # -----------------
        x = self.fc1(x)
        x = self.act(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        # -----------------
        y = self.fc2(x)

        return y

class Linear_1L_hetero(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid, pdrop):
        super(Linear_1L_hetero, self).__init__()

        self.pdrop = pdrop

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, 2 * output_dim)

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)

    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        # -----------------
        x = self.fc1(x)
        x = self.act(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        # -----------------
        y = self.fc2(x)

        return y

class Linear_2L(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(Linear_2L, self).__init__()

        self.pdrop = 0.5

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, output_dim)

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)

    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x = self.fc1(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        # -----------------
        x = self.act(x)
        # -----------------
        x = self.fc2(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        # -----------------
        x = self.act(x)
        # -----------------
        y = self.fc3(x)

        return y

    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)

        for i in range(Nsamples):
            y = self.forward(x, sample=True)
            predictions[i] = y

        return predictions

class MC_drop_net(BaseNet):
    eps = 1e-6

    def __init__(self, lr=1e-3, channels_in=3, side_in=28, cuda=True, classes=10, batch_size=128, weight_decay=0, n_hid=1200):
        super(MC_drop_net, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.channels_in = channels_in
        self.weight_decay = weight_decay
        self.classes = classes
        self.n_hid = n_hid
        self.batch_size = batch_size
        self.side_in = side_in
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = Linear_2L(input_dim=self.channels_in * self.side_in * self.side_in, output_dim=self.classes,
                               n_hid=self.n_hid)
        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5,
                                         weight_decay=self.weight_decay)

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #         self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=10, last_epoch=-1)

    def fit(self, x, y):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        self.optimizer.zero_grad()

        out = self.model(x)
        loss = F.cross_entropy(out, y, reduction='sum')

        loss.backward()
        self.optimizer.step()

        # out: (batch_size, out_channels, out_caps_dims)
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model(x)

        loss = F.cross_entropy(out, y, reduction='sum')

        probs = F.softmax(out, dim=1).data.cpu()

        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model.sample_predict(x, Nsamples)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def all_sample_eval(self, x, y, Nsamples):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

    def get_weight_samples(self):
        weight_vec = []

        state_dict = self.model.state_dict()

        for key in state_dict.keys():

            if 'weight' in key:
                weight_mtx = state_dict[key].cpu().data
                for weight in weight_mtx.view(-1):
                    weight_vec.append(weight)

        return np.array(weight_vec)

class MC_drop_net_BH(BaseNet):
    eps = 1e-6

    def __init__(self, lr=1e-3, input_dim=13, cuda=True, output_dim=1, batch_size=128, weight_decay=0, n_hid=1200, momentum=0, pdrop=0.5):
        super(MC_drop_net_BH, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.input_dim = input_dim
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.n_hid = n_hid
        self.batch_size = batch_size
        self.momentum = momentum
        self.pdrop = pdrop
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = Linear_1L(input_dim=self.input_dim, output_dim=self.output_dim,
                               n_hid=self.n_hid, pdrop=self.pdrop)
        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #         self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=10, last_epoch=-1)

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()

        out = self.model(x)
        loss = F.mse_loss(out, y, reduction='sum')

        loss.backward()
        self.optimizer.step()

        return loss.data

    def eval(self, x, y, train=False, samples=1):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        loss = 0

        outputs = torch.zeros(x.shape[0], self.output_dim, samples).cuda()

        if samples == 1:
            out = self.model(x)
            loss = F.mse_loss(out, y, reduction='sum')
            outputs[:, :, 0] = out

        elif samples > 1:

            for i in range(samples):
                out = self.model(x, sample=True)
                loss_i = F.mse_loss(out, y, reduction='sum')
                loss = loss + loss_i
                outputs[:, :, i] = out

            loss /= samples

        mean = torch.mean(outputs, dim=2)
        std = torch.std(outputs, dim=2)
        mse = F.mse_loss(mean, y, reduction='sum')

        return loss.data, mse.data, mean.data, std.data

    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        out = self.model.sample_predict(x, Nsamples)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def all_sample_eval(self, x, y, Nsamples):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

    def get_weight_samples(self):
        weight_vec = []

        state_dict = self.model.state_dict()

        for key in state_dict.keys():

            if 'weight' in key:
                weight_mtx = state_dict[key].cpu().data
                for weight in weight_mtx.view(-1):
                    weight_vec.append(weight)

        return np.array(weight_vec)

class MC_drop_net_BH_homo(BaseNet):
    eps = 1e-6

    def __init__(self, lr=1e-3, input_dim=13, cuda=True, output_dim=1, batch_size=128, weight_decay=0, n_hid=1200, momentum=0, pdrop=0.5):
        super(MC_drop_net_BH_homo, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.input_dim = input_dim
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.n_hid = n_hid
        self.batch_size = batch_size
        self.momentum = momentum
        self.pdrop = pdrop
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = Linear_1L_homo(input_dim=self.input_dim, output_dim=self.output_dim,
                               n_hid=self.n_hid, pdrop=self.pdrop)
        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #         self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=10, last_epoch=-1)

    def log_gaussian_loss(self, output, target, sigma, no_dim):
        exponent = -0.5 * (target - output) ** 2 / sigma ** 2
        log_coeff = -no_dim * torch.log(sigma)

        return - (log_coeff + exponent).sum()

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()

        out = self.model(x)
        loss = self.log_gaussian_loss(out, y, self.model.log_noise.exp(), self.model.output_dim)

        loss.backward()
        self.optimizer.step()

        return loss.data

    def eval(self, x, y, train=False, samples=1):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        loss = 0

        outputs = torch.zeros(x.shape[0], self.output_dim, samples).cuda()

        if samples == 1:
            out = self.model(x)
            loss = self.log_gaussian_loss(out, y, self.model.log_noise.exp(), self.model.output_dim)
            outputs[:, :, 0] = out

        elif samples > 1:

            for i in range(samples):
                out = self.model(x, sample=True)
                loss_i = self.log_gaussian_loss(out, y, self.model.log_noise.exp(), self.model.output_dim)
                loss = loss + loss_i
                outputs[:, :, i] = out

            loss /= samples

        mean = torch.mean(outputs, dim=2)
        std = torch.std(outputs, dim=2)
        mse = F.mse_loss(mean, y, reduction='sum')

        return loss.data, mse.data, mean.data, std.data

class MC_drop_net_BH_hetero(BaseNet):
    eps = 1e-6

    def __init__(self, lr=1e-3, input_dim=13, cuda=True, output_dim=1, batch_size=128, weight_decay=0, n_hid=1200, momentum=0, pdrop=0.5):
        super(MC_drop_net_BH_hetero, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.input_dim = input_dim
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.n_hid = n_hid
        self.batch_size = batch_size
        self.momentum = momentum
        self.pdrop = pdrop
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = Linear_1L_hetero(input_dim=self.input_dim, output_dim=self.output_dim,
                               n_hid=self.n_hid, pdrop=self.pdrop)
        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #         self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=10, last_epoch=-1)

    def log_gaussian_loss(self, output, target, sigma, no_dim, sum_reduce=True):
        exponent = -0.5 * (target - output) ** 2 / sigma ** 2
        log_coeff = -no_dim * torch.log(sigma) - 0.5 * no_dim * np.log(2 * np.pi)

        if sum_reduce:
            return -(log_coeff + exponent).sum()
        else:
            return -(log_coeff + exponent)

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()

        out = self.model(x)
        loss = self.log_gaussian_loss(out[:, :1], y, out[:, 1:].exp(), self.model.output_dim)

        loss.backward()
        self.optimizer.step()

        return loss.data

    def eval(self, x, y, train=False, samples=1):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        loss = 0

        outputs = torch.zeros(x.shape[0], self.output_dim * 2, samples).cuda()

        if samples == 1:
            out = self.model(x)
            loss = self.log_gaussian_loss(out[:, :1], y, out[:, 1:].exp(), self.model.output_dim)
            outputs[:, :, 0] = out

        elif samples > 1:

            for i in range(samples):
                out = self.model(x, sample=True)
                loss_i = self.log_gaussian_loss(out[:, :1], y, out[:, 1:].exp(), self.model.output_dim)
                loss = loss + loss_i
                outputs[:, :, i] = out

            loss /= samples

        mean = torch.mean(outputs[:, :1, :], dim=2)
        std = torch.std(outputs[:, :1, :], dim=2)
        noise = torch.mean(outputs[:, 1:, :] ** 2, dim=2)
        mse = F.mse_loss(mean, y, reduction='sum')

        return loss.data, mse.data, mean.data, std.data, noise.data

