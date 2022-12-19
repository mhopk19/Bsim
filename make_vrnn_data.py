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
    

if __name__ == '__main__':
    # time period
    timestep = 0.1
    timesteps = 200

    data = bat.create_data(batch_timesteps = timesteps, total_time = 2000, dt = 0.1)
    print("data shape", data.shape)
    num_batches = data.shape[0]
    
    batch = 5
    
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
    prev_inputs = 10
    row = 0
    counter = 0
    batch_num = 0
    print("seq length", int(timesteps / prev_inputs))
    
    vrnn_data = np.zeros((num_batches, timesteps, 2 * prev_inputs + 1))

    fig, ax = plt.subplots(1,2)
    showed = False
    for j in range(num_batches):
        for i in range(timesteps):
            """
            actual_current = data[batch][0][i]
            measured_voltage = data[batch][1][i] + np.random.normal(0,0.1,1)[0] 
            
            time.append(time[-1] + timestep)
            current.append(actual_current)
            mes_voltage.append(measured_voltage)
            true_SoC.append(data[batch][2][i])
            """
            if (i > prev_inputs):
                vrnn_data[j][i][0:10] = data[j][0][i-10:i]
                vrnn_data[j][i][10:20] = data[j][1][i-10:i]
                vrnn_data[j][i][20] = data[j][2][i]
           
        plt.cla()
        ax[1].plot(range(timesteps),vrnn_data[j,:,9])
        ax[1].plot(range(timesteps),vrnn_data[j,:,19])
        ax[1].plot(range(timesteps),vrnn_data[j,:,20])    
        plt.pause(0.01) 
        
        print(vrnn_data[j])
        ax[0].set_title("iterated transformed data")
        ax[0].imshow(vrnn_data[j,:,:])
        if (showed == False):
            plt.pause(1)
            showed=True
        else:
            ax[0].imshow(vrnn_data[j,:,:])
            plt.pause(0.01)
        
            
            
    print("data shape", vrnn_data.shape)
    # transform data to seq x batch x len
    """
    ax[0].set_title("iterated transformed data")
    ax[1].set_title("est. mean")
    ax[2].set_title("est. std")
    ax[0].imshow(x[:-1,0,:])
    ax[1].imshow(estimation[:,0,:])
    ax[2].imshow(std_estimation[:,0,:])
    if (showed == False):
        plt.colorbar()
        plt.pause(1)
    """     
            
    np.save("vrnn_train_data", vrnn_data)