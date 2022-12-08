import numpy as np
import matplotlib.pyplot as plt
import math


Vs = 3
La = 0.35
Tl = 10
J = 0.02
base_flux = 3
k_te = 0.1
Ra = 5
B = 0.3

x = np.array([0,0,0]) # state vector: current, angular velocity, angular acceleration
duration = 1000 # time steps is 1000
dt = 0.001 # change in time is 1ms
# total time is 1s

v = np.ones((duration))


class pm_dc_motor():
    
    def __init__(self, Vs = 3, La = 0.35, Tl = 10, J = 0.02, base_flux = 3,
                    k_te = 0.1, Ra = 5, B = np.array([1/La, 0, 0])):
        self.Vs = Vs
        self.La = La
        self.Tl = Tl
        self.J = J
        self.base_flux = base_flux
        self.k_te = k_te
        self.Ra = Ra
        self.B = B
        self.x = np.array([0,0,0]) # motor i, rotation angle, rotation speed
        
        
    def create_A(self, x):
        A = np.array([[-Ra/La,      0,  -k_te/La],
                      [0,           0,  1],
                      [k_te/J,      0,  -B/J]])
        return A
        
    def predict(self, input, initial_state = np.array([0,0,0]) ):
        x = initial_state
        x_hist = x
        for i in range(len(input)):
            A = create_A(x)
            dx = A @ x + self.B * input[i]
            x = x + dx * dt
            x_hist = np.vstack((x_hist,x))
            
        return x_hist


def create_A(x):
    A = np.array([[-Ra/La,      0,  -k_te/La],
                  [0,           0,  1],
                  [k_te/J,      0,  -B/J]])
    return A


"""
x_hist = x
for i in range(duration):
    A = create_A(x)
    dx = A @ x + np.array([1/La, 0, 0]) * v[i]
    x = x + dx * dt
    x_hist = np.vstack((x_hist,x))
"""

mtr = pm_dc_motor()
x_hist = mtr.predict(v, x)

plt.plot(range(len(x_hist[:,0])), x_hist[:,0], 'b')
plt.plot(range(len(x_hist[:,0])), x_hist[:,1], 'r')
plt.plot(range(len(x_hist[:,0])), x_hist[:,2], 'k')
plt.title("Permanent Magnet DC Motor Model Transient")
plt.ylabel("[A]/[rad]/[rad/s]")
plt.legend([r'$i_a(t)$',r'$\phi(t)$',r'$\omega(t)$'])
plt.xlabel("Time (ms)")
plt.show()

