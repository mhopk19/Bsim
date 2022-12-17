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

class CNNEncoder(nn.Module):
    def __init__(self):
        self.showing = True
        # inherit from nn.Module base class
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=2, stride=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(495, 50)
        self.fc2 = nn.Linear(50, 3)
        
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
        
    def to_buffer(self, v, i):
        self.buffer = torch.vstack((self.buffer,torch.tensor([v,i])))
        if (self.buffer.shape[0] > self.buffer_size):
            self.buffer = self.buffer[1:,:]
        print(self.buffer.shape)
        print(self.buffer)
        
    def encode(self):
        x = self.Encoder(self.buffer.unsqueeze(dim=0).float())
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