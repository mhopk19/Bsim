import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import numpy as np
import time


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

# changing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.finfo(torch.float).eps # numerical logs

#fig, ax = plt.subplots(1,4)
showed = False

class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.plotting = True

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.predict_dim = 2
        predict_dim = self.predict_dim
        self.n_layers = n_layers

        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim-1, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, predict_dim),
            nn.Softplus())
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, predict_dim))#,
            #nn.Sigmoid())

        #recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

    def forward(self, x):
        return self.predict_forward(x)
    
    def predict_forward(self, x):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0

        estimation = np.zeros((x.shape[0],x.shape[1],2))
        std_estimation = np.zeros((x.shape[0],x.shape[1],2))

        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=device)
        for t in range(1,x.size(0)):

            phi_x_t = self.phi_x(x[t-1,:,:20])

            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            #recurrence 
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # calculate the estimation
            estimation[t,:,] = dec_mean_t.data
            std_estimation[t,:,] = dec_std_t.data

            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            #nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            nll_loss += self._kld_gauss(dec_mean_t, dec_std_t, x[t,:,19:], 0.1 * torch.ones((dec_mean_t.shape)).float())

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)
        
        
        # show what the network is actuall calculating
        if (self.plotting):
            global showed
            if (np.random.normal() < 0.01):
                """
                ax[0].set_title("iterated transformed data")
                ax[1].set_title("est. mean")
                ax[2].set_title("est. std")
                ax[0].imshow(x[:-1,0,:])
                ax[1].imshow(estimation[:,0,:])
                ax[2].imshow(std_estimation[:,0,:])
                """
                
                # plot trajectory differences
                plt.cla()
                voltage_avg = estimation[:,0,0]
                voltage_min = voltage_avg - std_estimation[:,0,0]
                voltage_max = voltage_avg + std_estimation[:,0,0]
                soc_avg = estimation[:,0,1]
                soc_min = soc_avg - std_estimation[:,0,1]
                soc_max = soc_avg + std_estimation[:,0,1]            
                plt.plot(range(x.shape[0]-1),x[:-1,0,19],'r')
                plt.plot(range(x.shape[0]),voltage_avg,'r--')
                plt.fill_between(range(x.shape[0]), voltage_max, voltage_min, color = 'red', alpha = 0.5)
                plt.plot(range(x.shape[0]-1),x[:-1,0,20],'b')
                plt.plot(range(x.shape[0]),soc_avg,'b--')
                plt.fill_between(range(x.shape[0]), soc_max, soc_min, color = 'blue', alpha = 0.5)
                plt.legend(["actual voltage","est. voltage","volt. est. range", "actual soc","est. soc","soc est. range"])
                plt.title("Voltage and SOC trajectory prediction")
                
                if (showed == False):
                    #plt.colorbar()
                    plt.pause(40)
                    showed = True
                plt.pause(0.01)
        

        return kld_loss, nll_loss, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std) 
    
        
    def predict(self, x, reset = True):
        estimation = np.zeros((x.shape[0],x.shape[1],2))
        std_estimation = np.zeros((x.shape[0],x.shape[1],2))

        if (reset == True or (hasattr(self, 'h') == False)):
            self.h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=device)

        for t in range(x.size(0)):

            phi_x_t = self.phi_x(x[t-1,:,:20])

            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, self.h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            #prior
            prior_t = self.prior(self.h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, self.h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            #recurrence 
            _, self.h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), self.h)

            # calculate the estimation
            estimation[t,:,] = dec_mean_t.data
            std_estimation[t,:,] = dec_std_t.data
        
        return estimation, std_estimation

  

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim, device=device)

        h = torch.zeros(self.n_layers, 1, self.h_dim, device=device)
        for t in range(seq_len):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + torch.log(2*torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))
