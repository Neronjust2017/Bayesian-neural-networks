import math

from src.priors import *
from src.base_net import *

import torch.nn.functional as F
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, type_in="pred_intervals", alpha=0.1,
                 loss_type='qd_soft', censor_R=False,
		         soften=100., lambda_in=10., sigma_in=0.5, use_cuda=True):
        super().__init__()
        self.alpha = alpha
        self.lambda_in = lambda_in
        self.soften = soften
        self.loss_type = loss_type
        self.type_in = type_in
        self.censor_R = censor_R
        self.sigma_in = sigma_in
        if use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, y_pred, y_true):

        # compute loss

        if self.type_in == "pred_intervals":

            metric = []
            metric_name = []

            # get components
            y_U = y_pred[:, 0]
            y_L = y_pred[:, 1]
            y_T = y_true[:, 0]

            # set inputs and constants
            N_ = y_T.shape[0]
            alpha_ = self.alpha
            lambda_ = self.lambda_in

            # N_ = torch.tensor(y_T.shape[0])
            # alpha_ = torch.tensor(self.alpha)
            # lambda_ = torch.tensor(self.lambda_in)

            # in case want to do point predictions
            y_pred_mean = torch.mean(y_pred, dim=1)
            MPIW = torch.mean(y_U - y_L)

            # soft uses sigmoid
            gamma_U = torch.sigmoid((y_U - y_T) * self.soften)
            gamma_L = torch.sigmoid((y_T - y_L) * self.soften)
            gamma_ = torch.mul(gamma_U, gamma_L)
            ones_ = torch.ones_like(gamma_)

            # hard uses sign step fn
            zeros = torch.zeros_like(y_U)
            gamma_U_hard = torch.max(zeros, torch.sign(y_U - y_T))
            gamma_L_hard = torch.max(zeros, torch.sign(y_T - y_L))
            gamma_hard = torch.mul(gamma_U_hard, gamma_L_hard)

            # lube - lower upper bound estimation
            qd_lhs_hard = torch.div(torch.mean(torch.abs(y_U - y_L) * gamma_hard), torch.mean(gamma_hard) + 0.001)
            qd_lhs_soft = torch.div(torch.mean(torch.abs(y_U - y_L) * gamma_),
                                    torch.mean(gamma_) + 0.001)  # add small noise in case 0
            PICP_soft = torch.mean(gamma_)
            PICP_hard = torch.mean(gamma_hard)

            zero = torch.tensor(0.).to(self.device)
            qd_rhs_soft = lambda_ * math.sqrt(N_) * torch.pow(torch.max(zero, (1. - alpha_) - PICP_soft), 2)
            qd_rhs_hard = lambda_ * math.sqrt(N_) * torch.pow(torch.max(zero, (1. - alpha_) - PICP_hard), 2)

            # old method
            qd_loss_soft = qd_lhs_hard + qd_rhs_soft  # full LUBE w sigmoid for PICP
            qd_loss_hard = qd_lhs_hard + qd_rhs_hard  # full LUBE w step fn for PICP

            umae_loss = 0  # ignore this

            # gaussian log likelihood
            # already defined output nodes
            # y_U = mean, y_L = variance
            y_mean = y_U

            # from deep ensemble paper

            y_var_limited = torch.min(y_L, torch.tensor(10.).to(self.device))  # seem to need to limit otherwise causes nans occasionally
            y_var = torch.max(torch.log(1. + torch.exp(y_var_limited)), torch.tensor(10e-6).to(self.device))

            # to track nans
            self.y_mean = y_mean
            self.y_var = y_var

            gauss_loss = torch.log(y_var) / 2. + torch.div(torch.pow(y_T - y_mean, 2), 2. * y_var)  # this is -ve already
            gauss_loss = torch.mean(gauss_loss)
            # use mean so has some kind of comparability across datasets
            # but actually need to rescale and add constant if want to get actual results

            # set main loss type
            if self.loss_type == 'qd_soft':
                loss = qd_loss_soft
            elif self.loss_type == 'qd_hard':
                loss = qd_loss_hard
            # elif self.loss_type == 'umae_R_cens':
            #     loss = umae_loss_cens_R
            elif self.loss_type == 'gauss_like':
                loss = gauss_loss
            elif self.loss_type == 'picp':  # for loss visualisation
                loss = PICP_hard
            elif self.loss_type == 'mse':
                loss = torch.mean(torch.pow(y_U - y_T, 2))

            # add metrics
            u_capt = torch.mean(gamma_U_hard)  # apparently is quicker if define these
            l_capt = torch.mean(gamma_L_hard)  # here rather than in train loop

            # all_capt = torch.mean(gamma_hard)
            PICP = torch.mean(gamma_hard)

            # metric.append(u_capt)
            # metric_name.append('U capt.')
            # metric.append(l_capt)
            # metric_name.append('L capt.')
            metric.append(PICP)
            metric_name.append('PICP')
            metric.append(MPIW)
            metric_name.append('MPIW')
            # metric.append(tf.reduce_mean(tf.pow(y_T - y_pred_mean,2)))
            # metric_name.append("MSE mid")

        return loss, PICP, MPIW

class Linear_1L(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(Linear_1L, self).__init__()

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

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x = self.fc1(x)
        x = self.act(x)
        # -----------------
        y = self.fc2(x)

        return y

class QD_net_BH(BaseNet):
    eps = 1e-6

    def __init__(self, lr=1e-2, input_dim=13, cuda=True, output_dim=1, batch_size=128,
                 type_in="pred_intervals", alpha=0.1,loss_type='qd_soft', censor_R=False,
		         soften=100., lambda_in=10., sigma_in=0.5, bias_rand=False, out_biases=[2.,-2.],
                 weight_decay=0, n_hid=1200, momentum=0):
        super(QD_net_BH, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.type_in = type_in  # run some validation for these
        self.alpha = alpha
        self.loss_type = loss_type
        self.censor_R = censor_R
        self.sigma_in = sigma_in
        self.bias_rand = bias_rand
        self.soften = soften
        self.lambda_in = lambda_in
        self.out_biases = out_biases
        self.weight_decay = weight_decay
        self.n_hid = n_hid
        self.batch_size = batch_size
        self.momentum = momentum
        self.create_net()
        self.create_opt()
        self.create_loss()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = Linear_1L(input_dim=self.input_dim, output_dim=self.output_dim,
                               n_hid=self.n_hid)
        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                                  weight_decay=0)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
        #                                  weight_decay=self.weight_decay)

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #         self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=10, last_epoch=-1)
    def create_loss(self):
        self.loss = Loss(type_in=self.type_in, alpha=self.alpha,
                 loss_type=self.loss_type, censor_R=self.censor_R,
		         soften=self.soften, lambda_in=self.lambda_in,
                 sigma_in=self.sigma_in, use_cuda=self.cuda)

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()

        out = self.model(x)

        loss, PICP, MPIW = self.loss(out, y)

        loss.backward()
        self.optimizer.step()

        return loss.data, PICP.data, MPIW.data


    def eval(self, x, y, train=False, samples=1):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        out = self.model(x)

        loss, PICP, MPIW = self.loss(out, y)

        return loss.data, PICP.data, MPIW.data, out.data





