import numpy as np
import matplotlib.pyplot as plt
import math


R_ts = 0.78
C_ts = 461
R_tl = 15.5
C_tl = 2141
R_i = 0.00038

x = np.array([0,0,1]) # state vector: current, angular velocity, angular acceleration
duration = 10000 # time steps is 1000
dt = 0.001 # change in time is 1ms
# total time is 1s

C_capacity = 2600

period = int(duration/4)
i_batt = 0.5*np.ones((period))
i_batt = np.hstack((i_batt, np.zeros((period))))
i_batt = np.hstack((i_batt, np.zeros((period))))
i_batt = np.hstack((i_batt, -0.5*np.ones((period))))


def VOC(soc):
    a0 = -.635
    a1 = 26.8
    a2 = .31
    a3 = -2.02
    a4 = -1.27
    a5 = 7.21
    return a0 * math.exp(-a1*soc) + a2*math.pow(soc,3) + a3*math.pow(soc,2) + a4*soc + a5
    
    

def create_A(x):
    A = np.array([[-1/(R_ts * C_ts) ,       0,              0],
                  [0                ,       -1/(R_tl*C_tl), 0],
                  [0                ,       0,              0]])
    return A

v_batt = np.array([4.2])
x_hist = x
for i in range(duration):
    A = create_A(x)
    dx = A @ x + np.array([1/C_ts, 1/C_tl, 1/C_capacity]) * i_batt[i]
    x = x + dx * dt
    x_hist = np.vstack((x_hist,x))
    v_batt = np.vstack((v_batt, VOC(x[2]) - R_i*i_batt[i] - x[0] - x[1] ))


plt.plot(range(len(x_hist[:,0])), x_hist[:,0], 'b')
plt.plot(range(len(x_hist[:,0])), x_hist[:,1], 'r')
plt.plot(range(len(x_hist[:,0])), x_hist[:,2], 'y')
plt.plot(range(len(i_batt)), i_batt, 'k')
plt.plot(range(len(v_batt)), v_batt, 'g')
plt.title("Battery Equivalent Circuit ")
plt.ylabel("[A]/[V]/[0-1%]")
plt.legend([r'$V_{ts}$',r'$V_{tl}$',"SOC",r'$i_{batt}$',r'$v_{batt}$'])
plt.xlabel("Time (ms)")
plt.show()

