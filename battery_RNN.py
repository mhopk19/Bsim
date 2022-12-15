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

class LSTM(nn.Module):
    def __init__(self, hidden_layers=64, n_features = 3):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(n_features, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 3)
        
    def forward(self, y, future_preds=0):
        outputs, num_samples = [], y.size(0)
        h_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        
        """
        print("y shape", y.shape)
        print("amount of splits", len(y.split(self.n_features,dim=1)))
        print(" y split", y.split(self.n_features,dim=1)[0].shape, y.split(1,dim=1)[1].shape)
        """
        
        for time_step in y.split(self.n_features, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(time_step, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            outputs.append(output)
            
        for i in range(future_preds):
            # this only generates future predictions if we pass in future_preds>0
            # mirrors the code above, using last output/prediction as input
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
            
        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    def prediction(self, test_input, test_target, loss_fn, pred_L):
        with torch.no_grad():
            # remember the default output shape is timesteps x features flatten
            print("pred L", pred_L)
            pred = model(test_input, future_preds=pred_L)
            """
            print("prediction shape", pred.shape) 
            print("compared prediction shape", pred[:, :-pred_L*self.n_features].shape)
            print("test target shape", test_target.shape)
            """
            
            # use all pred samples, but only go to 999
            loss = loss_fn(pred[:, :-pred_L*self.n_features], test_target)
            y = pred.detach().numpy()
        return y
    

def training_loop(n_epochs, model, optimiser, loss_fn, 
                  train_input, train_target, test_input, test_target, pred_L = 100):
    for i in range(n_epochs):
        print("epoch", i ,"(time)", time.time())
        def closure():
            optimiser.zero_grad()
            out = model(train_input)
            loss = loss_fn(out, train_target)
            loss.backward()
            return loss
        optimiser.step(closure)
        
        
        y = model.prediction(test_input, test_target, loss_fn, pred_L)
        """
        with torch.no_grad():
            future = L
            pred = model(test_input, future_preds=future)
            # use all pred samples, but only go to 999
            loss = loss_fn(pred[:, :-future], test_target)
            y = pred.detach().numpy()
        """
        
        # draw figures
        plt.figure(figsize=(12,6))
        plt.title(f"Step {i+1}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = math.floor(train_input.shape[1]/model.n_features) # 999
        print("n",n)
        print("y",y.shape)
        print()
        def draw(yi):
            # reshaping inverse of Fortran indexing (used for LSTM)
            yi = np.reshape(yi, (model.n_features,-1), order = 'F')
            #yi = yi.reshape(model.n_features,-1)
            plt.plot(np.arange(n), yi[0][:n], 'r', linewidth=2.0)
            plt.plot(np.arange(n), yi[1][:n], 'g', linewidth=2.0)
            plt.plot(np.arange(n), yi[2][:n], 'b', linewidth=2.0)
            plt.plot(np.arange(n, n+pred_L), yi[0][n:], 'r:', linewidth=2.0)
            plt.plot(np.arange(n, n+pred_L), yi[1][n:], 'g:', linewidth=2.0)
            plt.plot(np.arange(n, n+pred_L), yi[2][n:], 'b:', linewidth=2.0)
        draw(y[int(random.random()) * y.shape[0]])
        plt.savefig("predict%d.png"%i, dpi=200)
        plt.close()
        # print the loss
        out = model(train_input)
        loss_print = loss_fn(out, train_target)
        print("Step: {}, Loss: {}".format(i, loss_print))



if __name__ == "__main__":
    load = True
    save = False
    
    if (load):
        data = (np.load("bat_input.npy"), np.load("bat_output.npy"))
    else:    
        data = bat.create_data(100, 200000, 0.1)
        data = bat.make_rnn_data(data)
    
    if (save):
        np.save("bat_input", data[0])
        np.save("bat_output", data[1])
        
    # data is a tuple object of input and output
    print("rnn input shape", data[0].shape)
    print("rnn output shape", data[1].shape)
    
    """
    train_input = torch.from_numpy(y[3:, :-1]) # (97, 999)
    train_target = torch.from_numpy(y[3:, 1:]) # (97, 999)
    
    test_input = torch.from_numpy(y[:3, :-1]) # (3, 999)
    test_target = torch.from_numpy(y[:3, 1:]) # (3, 999)
    """
    print("data input shape", data[0][3:].shape)
    print("data output shape", data[1][3:].shape)
    data_sequences = data[0].shape[0]
    data_features = data[0].shape[1]
    data_timesteps = data[0].shape[2]
    print("data sequences",data_sequences)
    
    
    test_sequences = 10
    
    # use Fortran like indexing to partition the features together
    
    train_input = torch.from_numpy(np.reshape(data[0][test_sequences:], (data_sequences - test_sequences,-1), order = 'F')).float()
    train_target = torch.from_numpy(np.reshape(data[1][test_sequences:], (data_sequences - test_sequences,-1), order = 'F')).float()
    
    test_input = torch.from_numpy(np.reshape(data[0][:test_sequences], (test_sequences, -1), order = 'F')).float()
    test_target = torch.from_numpy(np.reshape(data[1][:test_sequences], (test_sequences,-1), order = 'F')).float()

    # sample plot
    sample  = train_input[800]
    print("sample", sample)
    print("sample shape", sample.shape)
    plt.plot(range(sample.shape[0]), sample[:], 'r')
    plt.title("Plot of Reshaped sample to be fed into LSTM (ISSUE)")
    plt.show()
    
    orig_sample = np.reshape(sample.numpy(), (3,-1), order = 'F')
    plt.plot(range(orig_sample.shape[1]), orig_sample[1], 'r')
    plt.title("Plot of Reshaped sample to be fed into LSTM (ISSUE)")
    plt.show()
    

    model = LSTM()
    criterion = nn.MSELoss()
    optimiser = optim.LBFGS(model.parameters(), lr=0.08) 
    
    training_loop(30, model, optimiser, criterion, train_input, train_target, test_input, test_target, pred_L = 100)