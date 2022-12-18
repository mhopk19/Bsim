import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import battery_core as bat
import time
import math
import random
import cv2

EPS = torch.finfo(torch.float).eps # numerical logs
def kld_gauss(self, mean_1, std_1, mean_2, std_2):
    """Using std to compute KLD"""

    kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + 
        (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
        std_2.pow(2) - 1)
    return    0.5 * torch.sum(kld_element)

class CNNEncoder(nn.Module):
    def __init__(self, h_dim = 3):
        self.showing = True
        # inherit from nn.Module base class
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=2, stride=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(495, 50)
        self.fc2 = nn.Linear(50, z_dim)
        
    def forward(self, x):
        # showing variaible is only used for displaying the transformations from input to output
        showing = self.showing
        # propagate x through model adding non-linear activations across layers 
        # and display how the shape of the input changes across layers
        # this will only be shown the first time the neural network is called
        if showing: print("initial shape of input x:",x.shape)
        x = self.conv1(x)
        if showing: print("x after drop out:",x.shape)
        x = F.relu(x)
        if showing: print("x after non-linear activation:",x.shape)
        x = x.view(-1, 495)
        if showing: print("x after permutation:",x.shape)
        x = self.fc1(x)
        if showing: print("x after fully connected layer 1:",x.shape)
        x = F.relu(x)
        if showing: print("x after non-linear activation:",x.shape)
        # dropout layer for vector data
        x = F.dropout(x)
        if showing: print("dropout layer:",x.shape)
        x = self.fc2(x)
        
        if showing: print("outputs:", x[0])
        if (showing):
            self.showing = False
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))
    
class dSSM():
    def __init__(self):
        self.buffer = torch.empty((0,2), dtype=torch.float64)
        self.buffer_size = 100
        self.Encoder = CNNEncoder()
        
        # Encoder
        h_dim = 5 # must be able to  hold information on vts, vtl soc among other hidden info (electrochemical)  
        z_dim = 3 
        self.enc = CNNEncoder(h_dim = h_dim)
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        
        
    def likelihood_x(self, z_mean, z_std):
        # ocv linear function of soc is 3.12 + 1.16x
        soc_offset = torch.tensor([0, 0, 3.12])
        x_mean = z_mean + soc_offset
        
        matrix_transform = torch.tensor([[-1, 0, 0],
                                         [0, -1, 0],
                                         [0, 0, 3.12]])
        self.px = x_mean @ matrix_transform
        x_std = (matrix_transform @ matrix_transform).sqrt() @ z_std
        
        return x_std, z_std
    
    def forward(self, x):
        pass
        """
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0

        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=device)
        for t in range(x.size(0)):

            #encoder
            enc_out = self.Encoder(x)
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

            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            #nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, nll_loss, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std)
        """
        
    def to_buffer(self, v, i):
        self.buffer = torch.vstack((self.buffer,torch.tensor([v,i])))
        if (self.buffer.shape[0] > self.buffer_size):
            self.buffer = self.buffer[1:,:]
        print(self.buffer.shape)
        print(self.buffer)
        
    def encode(self):
        x = self.enc(self.buffer.unsqueeze(dim=0).float())
        print("x", x)
        return x
    


if __name__ == '__main__':
    # time period
    timestep = 0.1
    timesteps = 200

    data = bat.create_data(batch_timesteps = timesteps, total_time = 2000, dt = 0.1)
    print("data shape", data.shape)

    batch = 5
    
    dssm = dSSM()
    
    time         = [0]
    true_SoC = [data[batch][2][0]]
    true_voltage = [data[batch][1][0]]
    mes_voltage = [data[batch][1][0] + np.random.normal(0,0.1,1)[0]]
    current = [data[batch][0][0]]
    
    """
    update these based on data ^^
    true_SoC     = [battery_simulation.state_of_charge]
    estim_SoC    = [Kf.x[0,0]]
    true_voltage = [battery_simulation.voltage]
    mes_voltage  = [battery_simulation.voltage + np.random.normal(0,0.1,1)[0]]
    current      = [battery_simulation.current]
    """

    for i in range(timesteps):
        actual_current = data[batch][0][i]
        measured_voltage = data[batch][1][i] + np.random.normal(0,0.1,1)[0] 
        
        dssm.to_buffer(measured_voltage, actual_current)
        
        time.append(time[-1] + timestep)
        current.append(actual_current)
        mes_voltage.append(measured_voltage)
        true_SoC.append(data[batch][2][i])
        
    dssm.encode()
    
    plt.plot(range(timesteps+1), true_SoC,'g')
    plt.plot(range(timesteps+1), current,'k')
    plt.show()