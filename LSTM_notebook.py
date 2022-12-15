import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

N = 50 # number of samples
L = 300 # length of each sample (number of values for each sine wave)
T = 20 # width of the wave

x = np.empty((N,L), np.float32) # instantiate empty array
random_shift = np.random.randint(-4*T, 4*T, N).reshape(N,1)
x[:] = np.arange(L) + random_shift
y = np.sin(x/1.0/T).astype(np.float32)


# every row of x is a list of positions x in the sin function, where each row is shifted by an amount
# y is the output of the sin function, sin(x) at every point x


class LSTM(nn.Module):
    def __init__(self, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)
        
    def forward(self, y, future_preds=0):
        outputs, num_samples = [], y.size(0)
        h_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        
        print("y shape", y.shape)
        print("amount of splits", len(y.split(1,dim=1)))
        print(" y split", y.split(1,dim=1)[0].shape, y.split(1,dim=1)[1].shape)
        for time_step in y.split(1, dim=1):
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
            pred = model(test_input, future_preds=pred_L)
            # use all pred samples, but only go to 999
            loss = loss_fn(pred[:, :-pred_L], test_target)
            y = pred.detach().numpy()
        return y
    
    
train_input = torch.from_numpy(y[3:, :-1]) # (97, 999)
train_target = torch.from_numpy(y[3:, 1:]) # (97, 999)

test_input = torch.from_numpy(y[:3, :-1]) # (3, 999)
test_target = torch.from_numpy(y[:3, 1:]) # (3, 999)

model = LSTM()
criterion = nn.MSELoss()
optimiser = optim.LBFGS(model.parameters(), lr=0.08)

def training_loop(n_epochs, model, optimiser, loss_fn, 
                  train_input, train_target, test_input, test_target, pred_L = 100):
    for i in range(n_epochs):
        print("epoch", i)
        def closure():
            optimiser.zero_grad()
            print("train input shape", train_input.shape)
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
        n = train_input.shape[1] # 999
        def draw(yi, colour):
            plt.plot(np.arange(n), yi[:n], colour, linewidth=2.0)
            plt.plot(np.arange(n, n+pred_L), yi[n:], colour+":", linewidth=2.0)
        draw(y[0], 'r')
        draw(y[1], 'b')
        draw(y[2], 'g')
        plt.savefig("predict%d.png"%i, dpi=200)
        plt.close()
        # print the loss
        out = model(train_input)
        loss_print = loss_fn(out, train_target)
        print("Step: {}, Loss: {}".format(i, loss_print))

training_loop(40, model, optimiser, criterion, train_input, train_target, test_input, test_target, pred_L = L)
        
