from src.priors import *

from src.base_net import *

import torch.nn.functional as F
import torch.nn as nn
import copy
import math

def calculate_kl(log_alpha):
    return 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))

class VdLinear(nn.Module):
    """
    variational dropout

    """
    def __init__(self, n_in, n_out, alpha_shape=(1, 1), bias=True):
        super(VdLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.alpha_shape = alpha_shape
        self.bias = bias

        # Learnable parameters -> Initialisation is set empirically.
        self.W = nn.Parameter(torch.Tensor(self.n_out, self.n_in))
        self.log_alpha = nn.Parameter(torch.Tensor(*self.alpha_shape))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.n_out))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.kl_value = calculate_kl

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, X, sample=False):

            mean = F.linear(X, self.W)
            if self.bias is not None:
                mean = mean + self.bias

            sigma = torch.exp(self.log_alpha) * self.W * self.W

            std = torch.sqrt(1e-16 + F.linear(X * X, sigma))

            if self.training or sample:
                epsilon = std.data.new(std.size()).normal_()
            else:
                epsilon = 0.0

            # Local reparameterization trick
            out = mean + std * epsilon

            kl = self.kl_loss()

            return out, kl

    def kl_loss(self):
        return self.W.nelement() * self.kl_value(self.log_alpha) / self.log_alpha.nelement()

class vd_linear_1L(nn.Module):
    """1 hidden layer Variational Dropout Network"""
    def __init__(self, input_dim, output_dim, alpha_shape=(1, 1), bias=True, n_hid=50):
        super(vd_linear_1L, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha_shape = alpha_shape
        self.bias = bias

        self.bfc1 = VdLinear(input_dim, n_hid, self.alpha_shape, self.bias)
        self.bfc2 = VdLinear(n_hid, output_dim, self.alpha_shape, self.bias)

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)

    def forward(self, x, sample=False):
        tkl = 0.0

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x, kl = self.bfc1(x, sample)
        tkl = tkl + kl
        # -----------------
        x = self.act(x)
        # -----------------
        y, kl = self.bfc2(x, sample)
        tkl = tkl + kl

        return y, tkl

    def sample_predict(self, x, Nsamples):
        """Used for estimating the data's likelihood by approximately marginalising the weights with MC"""
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tkl_vec = np.zeros(Nsamples)

        for i in range(Nsamples):
            y, tkl = self.forward(x, sample=True)
            predictions[i] = y
            tkl_vec[i] = tkl

        return predictions, tkl_vec

class vd_linear_1L_homo(nn.Module):
    """1 hidden layer Variational Dropout Network"""
    def __init__(self, input_dim, output_dim, alpha_shape=(1, 1), bias=True, n_hid=50, init_log_noise=0):
        super(vd_linear_1L_homo, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha_shape = alpha_shape
        self.bias = bias

        self.bfc1 = VdLinear(input_dim, n_hid, self.alpha_shape, self.bias)
        self.bfc2 = VdLinear(n_hid, output_dim, self.alpha_shape, self.bias)

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)
        self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))

    def forward(self, x, sample=False):
        tkl = 0.0

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x, kl = self.bfc1(x, sample)
        tkl = tkl + kl
        # -----------------
        x = self.act(x)
        # -----------------
        y, kl = self.bfc2(x, sample)
        tkl = tkl + kl

        return y, tkl

    def sample_predict(self, x, Nsamples):
        """Used for estimating the data's likelihood by approximately marginalising the weights with MC"""
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tkl_vec = np.zeros(Nsamples)

        for i in range(Nsamples):
            y, tkl = self.forward(x, sample=True)
            predictions[i] = y
            tkl_vec[i] = tkl

        return predictions, tkl_vec

class vd_linear_1L_hetero(nn.Module):
    """1 hidden layer Variational Dropout Network"""
    def __init__(self, input_dim, output_dim, alpha_shape=(1, 1), bias=True, n_hid=50):
        super(vd_linear_1L_hetero, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha_shape = alpha_shape
        self.bias = bias

        self.bfc1 = VdLinear(input_dim, n_hid, self.alpha_shape, self.bias)
        self.bfc2 = VdLinear(n_hid, 2 * output_dim, self.alpha_shape, self.bias)

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)

    def forward(self, x, sample=False):
        tkl = 0.0

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x, kl = self.bfc1(x, sample)
        tkl = tkl + kl
        # -----------------
        x = self.act(x)
        # -----------------
        y, kl = self.bfc2(x, sample)
        tkl = tkl + kl

        return y, tkl

    def sample_predict(self, x, Nsamples):
        """Used for estimating the data's likelihood by approximately marginalising the weights with MC"""
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tkl_vec = np.zeros(Nsamples)

        for i in range(Nsamples):
            y, tkl = self.forward(x, sample=True)
            predictions[i] = y
            tkl_vec[i] = tkl

        return predictions, tkl_vec

class VD_Bayes_Net(BaseNet):

    eps = 1e-6

    def __init__(self, lr=1e-3, channels_in=3, side_in=28, cuda=True, classes=10, batch_size=128, Nbatches=0,
                 nhid=1200, alpha_shape=(1, 1), bias=True):
        super(VD_Bayes_Net, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.channels_in = channels_in
        self.classes = classes
        self.batch_size = batch_size
        self.Nbatches = Nbatches
        self.nhid = nhid
        self.side_in = side_in
        self.alpha_shape = alpha_shape
        self.bias = bias
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        # self.model = bayes_linear_2L(input_dim=self.channels_in * self.side_in * self.side_in,
        #                              output_dim=self.classes, n_hid=self.nhid, prior_instance=self.prior_instance)

        self.model = vd_linear_1L(input_dim=self.channels_in * self.side_in * self.side_in,
                                     output_dim=self.classes, n_hid=self.nhid, alpha_shape=self.alpha_shape, bias=self.bias)

        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0)

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #         self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=10, last_epoch=-1)

    def fit(self, x, y, samples=1):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        self.optimizer.zero_grad()

        if samples == 1:
            out, tkl = self.model(x)
            mlpdw = F.cross_entropy(out, y, reduction='sum')
            Edkl = tkl / self.Nbatches

        elif samples > 1:
            mlpdw_cum = 0
            Edkl_cum = 0

            for i in range(samples):
                out, tkl = self.model(x, sample=True)
                mlpdw_i = F.cross_entropy(out, y, reduction='sum')
                Edkl_i = tkl / self.Nbatches
                mlpdw_cum = mlpdw_cum + mlpdw_i
                Edkl_cum = Edkl_cum + Edkl_i

            mlpdw = mlpdw_cum / samples
            Edkl = Edkl_cum / samples

        loss = Edkl + mlpdw
        loss.backward()
        self.optimizer.step()

        # out: (batch_size, out_channels, out_caps_dims)
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return Edkl.data, mlpdw.data, err

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, _, _ = self.model(x)

        loss = F.cross_entropy(out, y, reduction='sum')

        probs = F.softmax(out, dim=1).data.cpu()

        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        """Prediction, only returining result with weights marginalised"""
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, _ = self.model.sample_predict(x, Nsamples)

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
        """Returns predictions for each MC sample"""
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, _, = self.model.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

class VD_Bayes_Net_BH(BaseNet):

    eps = 1e-6

    def __init__(self, lr=1e-3, input_dim=13, cuda=True, output_dim=1, batch_size=128, Nbatches=0,
                 nhid=1200, alpha_shape=(1, 1), bias=True, momentum=0):
        super(VD_Bayes_Net_BH, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.Nbatches = Nbatches
        self.nhid = nhid
        self.alpha_shape = alpha_shape
        self.bias = bias
        self.momentum = momentum
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = vd_linear_1L(input_dim=self.input_dim,output_dim=self.output_dim,
                                  n_hid=self.nhid, alpha_shape=self.alpha_shape, bias=self.bias)

        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #         self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=10, last_epoch=-1)

    def fit(self, x, y, samples=1):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()

        outputs = torch.zeros(x.shape[0], self.output_dim, samples).cuda()
        if samples == 1:
            out, tkl = self.model(x)
            mlpdw = F.mse_loss(out, y, reduction='sum')
            Edkl = tkl / self.Nbatches
            outputs[:,:,0] = out

        elif samples > 1:
            mlpdw_cum = 0
            Edkl_cum = 0

            for i in range(samples):
                out, tkl = self.model(x, sample=True)
                mlpdw_i = F.mse_loss(out, y, reduction='sum')
                Edkl_i = tkl / self.Nbatches
                mlpdw_cum = mlpdw_cum + mlpdw_i
                Edkl_cum = Edkl_cum + Edkl_i

                outputs[:, :, i] = out

            mlpdw = mlpdw_cum / samples
            Edkl = Edkl_cum / samples

        mean = torch.mean(outputs, dim=2)
        mse = F.mse_loss(mean, y, reduction='sum')

        loss = Edkl + mlpdw
        loss.backward()
        self.optimizer.step()

        return Edkl.data, mlpdw.data, mse.data

    def eval(self, x, y, train=False, samples=1):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        loss = 0

        outputs = torch.zeros(x.shape[0], self.output_dim, samples).cuda()

        if samples == 1:
            out, _= self.model(x)
            loss = F.mse_loss(out, y, reduction='sum')
            outputs[:,:,0] = out

        elif samples > 1:
            mlpdw_cum = 0

            for i in range(samples):
                out, _= self.model(x, sample=True)
                mlpdw_i = F.mse_loss(out, y, reduction='sum')
                mlpdw_cum = mlpdw_cum + mlpdw_i
                outputs[:,:,i] = out

            mlpdw = mlpdw_cum / samples
            loss = mlpdw

        mean = torch.mean(outputs, dim=2)
        std = torch.std(outputs, dim=2)
        mse = F.mse_loss(mean, y, reduction='sum')

        return loss.data, mse.data, mean.data, std.data

class VD_Bayes_Net_BH_homo(BaseNet):

    eps = 1e-6

    def __init__(self, lr=1e-3, input_dim=13, cuda=True, output_dim=1, batch_size=128, Nbatches=0,
                 nhid=1200, alpha_shape=(1, 1), bias=True, momentum=0):
        super(VD_Bayes_Net_BH_homo, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.Nbatches = Nbatches
        self.nhid = nhid
        self.alpha_shape = alpha_shape
        self.bias = bias
        self.momentum = momentum
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = vd_linear_1L_homo(input_dim=self.input_dim,output_dim=self.output_dim,
                                  n_hid=self.nhid, alpha_shape=self.alpha_shape, bias=self.bias)

        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #         self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=10, last_epoch=-1)

    def log_gaussian_loss(self, output, target, sigma, no_dim):
        exponent = -0.5 * (target - output) ** 2 / sigma ** 2
        log_coeff = -no_dim * torch.log(sigma)

        return - (log_coeff + exponent).sum()

    def fit(self, x, y, samples=1):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()

        outputs = torch.zeros(x.shape[0], self.output_dim, samples).cuda()
        if samples == 1:
            out, tkl = self.model(x)
            mlpdw = self.log_gaussian_loss(out, y, self.model.log_noise.exp(), self.model.output_dim)
            Edkl = tkl / self.Nbatches
            outputs[:,:,0] = out

        elif samples > 1:
            mlpdw_cum = 0
            Edkl_cum = 0

            for i in range(samples):
                out, tkl = self.model(x, sample=True)
                mlpdw_i = self.log_gaussian_loss(out, y, self.model.log_noise.exp(), self.model.output_dim)
                Edkl_i = tkl / self.Nbatches
                mlpdw_cum = mlpdw_cum + mlpdw_i
                Edkl_cum = Edkl_cum + Edkl_i

                outputs[:, :, i] = out

            mlpdw = mlpdw_cum / samples
            Edkl = Edkl_cum / samples

        mean = torch.mean(outputs, dim=2)
        mse = F.mse_loss(mean, y, reduction='sum')

        loss = Edkl + mlpdw
        loss.backward()
        self.optimizer.step()

        return Edkl.data, mlpdw.data, mse.data

    def eval(self, x, y, train=False, samples=1):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        loss = 0

        outputs = torch.zeros(x.shape[0], self.output_dim, samples).cuda()

        if samples == 1:
            out, _= self.model(x)
            loss = self.log_gaussian_loss(out, y, self.model.log_noise.exp(), self.model.output_dim)
            outputs[:,:,0] = out

        elif samples > 1:
            mlpdw_cum = 0

            for i in range(samples):
                out, _= self.model(x, sample=True)
                mlpdw_i = self.log_gaussian_loss(out, y, self.model.log_noise.exp(), self.model.output_dim)
                mlpdw_cum = mlpdw_cum + mlpdw_i
                outputs[:,:,i] = out

            mlpdw = mlpdw_cum / samples
            loss = mlpdw

        mean = torch.mean(outputs, dim=2)
        std = torch.std(outputs, dim=2)
        mse = F.mse_loss(mean, y, reduction='sum')

        return loss.data, mse.data, mean.data, std.data

class VD_Bayes_Net_BH_hetero(BaseNet):

    eps = 1e-6

    def __init__(self, lr=1e-3, input_dim=13, cuda=True, output_dim=1, batch_size=128, Nbatches=0,
                 nhid=1200, alpha_shape=(1, 1), bias=True, momentum=0):
        super(VD_Bayes_Net_BH_hetero, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.Nbatches = Nbatches
        self.nhid = nhid
        self.alpha_shape = alpha_shape
        self.bias = bias
        self.momentum = momentum
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = vd_linear_1L_hetero(input_dim=self.input_dim,output_dim=self.output_dim,
                                  n_hid=self.nhid, alpha_shape=self.alpha_shape, bias=self.bias)

        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #         self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=10, last_epoch=-1)

    def log_gaussian_loss(self, output, target, sigma, no_dim, sum_reduce=True):
        exponent = -0.5 * (target - output) ** 2 / sigma ** 2
        log_coeff = -no_dim * torch.log(sigma) - 0.5 * no_dim * np.log(2 * np.pi)

        if sum_reduce:
            return -(log_coeff + exponent).sum()
        else:
            return -(log_coeff + exponent)

    def fit(self, x, y, samples=1):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()

        outputs = torch.zeros(x.shape[0], self.output_dim * 2, samples).cuda()
        if samples == 1:
            out, tkl = self.model(x)
            mlpdw = self.log_gaussian_loss(out[:, :1], y, out[:, 1:].exp(), self.model.output_dim)
            Edkl = tkl / self.Nbatches
            outputs[:,:,0] = out

        elif samples > 1:
            mlpdw_cum = 0
            Edkl_cum = 0

            for i in range(samples):
                out, tkl = self.model(x, sample=True)
                mlpdw_i = self.log_gaussian_loss(out[:, :1], y, out[:, 1:].exp(), self.model.output_dim)
                Edkl_i = tkl / self.Nbatches
                mlpdw_cum = mlpdw_cum + mlpdw_i
                Edkl_cum = Edkl_cum + Edkl_i

                outputs[:, :, i] = out

            mlpdw = mlpdw_cum / samples
            Edkl = Edkl_cum / samples

        mean = torch.mean(outputs[:, :1, :], dim=2)
        mse = F.mse_loss(mean, y, reduction='sum')

        loss = Edkl + mlpdw
        loss.backward()
        self.optimizer.step()

        return Edkl.data, mlpdw.data, mse.data

    def eval(self, x, y, train=False, samples=1):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        loss = 0

        outputs = torch.zeros(x.shape[0], self.output_dim * 2, samples).cuda()

        if samples == 1:
            out, _= self.model(x)
            loss = self.log_gaussian_loss(out[:, :1], y, out[:, 1:].exp(), self.model.output_dim)
            outputs[:,:,0] = out

        elif samples > 1:
            mlpdw_cum = 0

            for i in range(samples):
                out, _= self.model(x, sample=True)
                mlpdw_i = self.log_gaussian_loss(out[:, :1], y, out[:, 1:].exp(), self.model.output_dim)
                mlpdw_cum = mlpdw_cum + mlpdw_i
                outputs[:,:,i] = out

            mlpdw = mlpdw_cum / samples
            loss = mlpdw

        mean = torch.mean(outputs[:, :1, :], dim=2)
        std = torch.std(outputs[:, :1, :], dim=2)
        noise = torch.mean(outputs[:, 1:, :]**2, dim=2)
        mse = F.mse_loss(mean, y, reduction='sum')

        return loss.data, mse.data, mean.data, std.data, noise.data