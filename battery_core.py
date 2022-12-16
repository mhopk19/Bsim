import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# empirical Poopanya et. al found parameters
def Rs(soc):
    # milli ohms
    tabley = np.array([56, 56, 45, 42, 43, 42, 43, 44, 42, 50, 50]) / 1000
    return np.interp(soc, np.array([0,10,20,30,40,50,60,70,80,90,100]), tabley)

def Rts(soc):
    # milli ohms
    tabley = np.array([13, 13, 20, 12, 15, 16, 17, 16, 17, 19, 19]) / 1000
    return np.interp(soc, np.array([0,10,20,30,40,50,60,70,80,90,100]), tabley)

def Rtl(soc):
    # milli ohms
    tabley = np.array([10, 10, 9, 9, 18, 13, 7, 10, 100, 10, 10]) / 1000
    return np.interp(soc, np.array([0,10,20,30,40,50,60,70,80,90,100]), tabley)

def Cts(soc):
    # kilo farads
    tabley = np.array([11, 11, 0.45, 0.5, 3, 7, 4, 3, 1, 3, 3]) * 1000
    return np.interp(soc, np.array([0,10,20,30,40,50,60,70,80,90,100]), tabley)

def Ctl(soc):
    # kilo farads
    tabley = np.array([5, 5, 79, 20, 19, 1, 42, 183, 5, 100, 100]) * 1000
    return np.interp(soc, np.array([0,10,20,30,40,50,60,70,80,90,100]), tabley)

def VOC(soc):
    # open circuit voltage from interpolation function
    tabley = np.array([2.81, 3.23, 3.45, 3.56, 3.65, 3.76, 3.84, 3.91, 4.08, 4.12, 4.2])
    ocv = np.interp(soc, np.array([0,10,20,30,40,50,60,70,80,90,100]), tabley)
    return ocv

def custom_empirical_parameters(soc,params):
    tablex = np.array([0,10,20,30,40,50,60,70,80,90,100])
    # rs
    rs_tabley = np.array(params[:11])
    rts_tabley = np.array(params[11:22])
    rtl_tabley = np.array(params[22:33])
    cts_tabley = np.array(params[33:44])
    ctl_tabley = np.array(params[44:55])
    voc_tabley = np.array(params[55:66])
    
    params = [np.interp(soc, tablex, rts_tabley),
              np.interp(soc, tablex, rtl_tabley),
              np.interp(soc, tablex, cts_tabley),
              np.interp(soc, tablex, ctl_tabley),
              np.interp(soc, tablex, rs_tabley),
              np.interp(soc, tablex, voc_tabley)]
    # rts, rtl, cts, ctl, rs, voc
    return params


"""
core 3200mAH 18650 Battery model with default parameters
and functions for updating state space parameters
"""
class battery18650():
    def __init__(self, start_voltage  = 4.2, start_soc = 100, battery_params = {}):
        self.Rs = Rs(start_soc)
        self.Rts = Rts(start_soc)
        self.Rtl = Rtl(start_soc)
        self.Cts = Cts(start_soc)
        self.Ctl = Ctl(start_soc)
        self.capacity = 3200
        # x state x [0] - vts, [1] - vtl, [2] - soc
        self.x = np.array([0,0,start_soc/100, start_voltage])
        self.x_hist = np.zeros((0,4))
        self.param_func = self.empirical_defaults
        self.dt = 0.1
        # loading battery params for dictionary
        if len(battery_params) > 0:
            self.load_battery_params(battery_params)
      
    def load_battery_params(self, dict):
        if "capacity" in dict.keys():
            self.capacity = dict["capacity"]
    
    def clear_data(self):
        self.x_hist = np.zeros((0,4))
      
    def empirical_defaults(self):
        # update params based on found empirical functions
        soc = self.x[2] * 100
        self.Rs = Rs(soc)
        self.Rts = Rts(soc)
        self.Rtl = Rtl(soc)
        self.Cts = Cts(soc)
        self.Ctl = Ctl(soc)
        self.Voc = VOC(soc)
        
    def update_params(self):
        if (self.param_func != None):
            self.param_func()
            
    def make_A(self):
        A = np.array([[-1/(self.Rts * self.Cts),       0,              0,   0],
                      [0                ,       -1/(self.Rtl*self.Ctl), 0,  0],
                      [0                ,       0,              0,      0],
                      [0                ,       0,              0,      0]])
        return A
    
    def step(self, i):
        # i is the applied current
        # update parameters based on scheme, make A matrix and then update states
        self.update_params()
        A = self.make_A()
        dx = A @ self.x + np.array([1/self.Cts, 1/self.Ctl, -1/self.capacity, 0]) * i
        self.x = self.x + dx * self.dt
        # update the battery voltage
        self.x[3] = self.Voc - self.Rs * i - self.x[0] - self.x[1]
        self.x_hist = np.vstack((self.x_hist, self.x))


def create_data(batch_timesteps = 100, total_time = 6000, dt = 0.1, cur_limits = (-1,1), 
                    battery_params_dict = {}, rnn_data = True):
    print("creating battery data...")
    battery = battery18650(battery_params = battery_params_dict)
    # battery cycling random walk state
    rw_state = np.array([0,0]) # 0 - frequency, 1 - amplitude
    num_batches = int(total_time / batch_timesteps)
    batched_data = np.zeros((num_batches, 3, batch_timesteps)) # 3 (i bat., v bat. soc) x time_steps x batches
    
    for batch in range(int(total_time / batch_timesteps)):
        # reset current vector
        i_batt = np.zeros((0,1))
        # reset dead battery at beginning of new batch
        if (battery.x[2] <= 0.05):
            battery = battery18650()
            
        for t in range(batch_timesteps):
            rw_state = rw_state + (1/math.log(batch_timesteps)) * np.sign(np.random.random(2) - np.array([0.5,0.5]))
            # randomly exit random path
            if (np.random.random() < 0.001):
                rw_state = np.array([0,0])
            new_value = rw_state[1] * np.sin(rw_state[0] * total_time)
            new_value = max(min(cur_limits[1],new_value),cur_limits[0])
            new_value = abs(new_value)
            # dont kill the battery
            if (battery.x[2] <= 0.05):
                new_value = 0
            i_batt = np.vstack((i_batt, new_value))
            
        for t in range(batch_timesteps):
            battery.step(i_batt[t])
        
        batch_data =  np.vstack((i_batt.T, battery.x_hist[:,3],battery.x_hist[:,2]))
        #print("batch data shape", batch_data.shape)
        #print("batched_data shape", batched_data.shape)
        batched_data[batch] = batch_data
        # clear battery data
        battery.clear_data()
    
    print("finished creating battery data [data shape] ", batched_data.shape)
    batched_data = batched_data.reshape((-1, 3, batch_timesteps))
    
    # batch data shape
    """
    batched data shape
    batch x (i_batt,v_batt,SOC) x time step
    """
        
    return batched_data
    
def make_rnn_data(data):
    """
    The input to this function is batched data 
    and the output is a set of inputs and outputs for rnn training
    we have to shift input with regards to output 
    offset the time sequences and also shuffle them
    """
    num_batches = data.shape[0]
    num_features = data.shape[1]
    num_timesteps = data.shape[2]
    rnn_input = np.empty((num_batches, num_features, num_timesteps - 1))
    rnn_output = np.empty((num_batches, num_features, num_timesteps - 1))
    for i in range(num_batches):
        rnn_input[i,:,:] = data[i,:,:-1]
        rnn_output[i,:,:] = data[i,:,:-1]
        
    return rnn_input, rnn_output

def draw_time_sequences(data, dt = 0.1):
    for batch_data in data:
        time_range = batch_data.shape[1]
        plt.plot(range(time_range), batch_data[0,:], 'b')
        plt.plot(range(time_range), batch_data[1,:], 'r')
        plt.plot(range(time_range), batch_data[2,:], 'y')
        plt.title("State-Space Time Sequences (Batches)")
        plt.ylabel("[A]/[V]/[0-1%]")
        plt.legend([r'$i_{batt}$',r'$v_{batt.}$',"SOC"])
        plt.xlabel("Timesteps ({}s)".format(dt))
        plt.draw()
        plt.pause(0.05)
        plt.cla()


if __name__ == "__main__":
    data = create_data(100, 200000, 0.1)
    
    print("data shape", data.shape)
    time.sleep(0.5)
    draw_time_sequences(data)
        
    data = make_rnn_data(data)
    print("rnn input shape", data[0].shape)
    print("rnn output shape", data[1].shape)
    
    
    """
    duration = 3000
    battery = battery18650()
    i_batt = np.random.random((duration, 1)) + 9 * np.ones((duration,1))
    for t in range(duration):
        battery.step(i_batt[t])
    
    
    print("final state of charge", battery.x[2])
    print("soc", battery.x_hist[:,2])
    time_range = battery.x_hist[:,0]
    plt.plot(range(len(time_range)), battery.x_hist[:,0], 'b')
    plt.plot(range(len(time_range)), battery.x_hist[:,1], 'r')
    plt.plot(range(len(time_range)), battery.x_hist[:,2], 'y')
    plt.plot(range(len(i_batt)), i_batt, 'k')
    plt.plot(range(len(time_range)), battery.x_hist[:,3], 'g')
    plt.title("Battery Equivalent Circuit ")
    plt.ylabel("[A]/[V]/[0-1%]")
    plt.legend([r'$V_{ts}$',r'$V_{tl}$',"SOC",r'$i_{batt}$',r'$v_{batt}$'])
    plt.xlabel("Timesteps ({}s)".format(battery.dt))
    plt.show()
    """