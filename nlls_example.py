"""
State-space trajectory system identification

"""

import scipy.optimize as sciopt
import numpy as np
import matplotlib.pyplot as plt

time_length = 100
initial_state = np.array([1,1])

def gen_traj(x):
    # x[0] freq x[1] amplitude
    state = np.expand_dims(initial_state, 0).T
    traj = state
    traj_data = traj
    print("inital traj", traj)
    A = np.matrix([[x[0], x[1]],[x[2],x[3]]])
    for i in range(time_length):
        traj_data = np.hstack((traj_data,A @ state))
        print("data", traj_data)
        state = traj_data[:,-1]
        print("new state", state)      
    return traj_data

A_components = [0.5, 0, 0, 0.1] # components of matrix A
data = gen_traj(A_components)
#data = data + np.random.random((time_length)) * 0.1
    
def loss(params):
    new_traj = gen_traj(params)
    print("new traj", new_traj)
    residual = np.power(np.ravel(new_traj[0,:] - data[0,:]),2)
    print("residual",residual)
    residual = np.hstack((residual, np.power(np.ravel(new_traj[1,:] - data[1,:]),2)) )
    #print("residual",residual)
    return residual
    
result = sciopt.least_squares(loss,(0.3,0.3,0.3,0.3),jac='3-point')

print("result", result)

fig, ax = plt.subplots(2,1)
ax[0].plot(range(time_length+1), np.ravel(data[0,:]), 'r')
ax[0].plot(range(time_length+1), np.ravel(gen_traj([result.x[0],result.x[1], result.x[2], result.x[3]])[0,:]), 'b')
ax[0].set_title("State-Space Least Squares Optimization for states 1 and 2")
ax[0].set_ylabel("state 1")
#
ax[1].plot(range(time_length+1), np.ravel(data[1,:]), 'r')
ax[1].plot(range(time_length+1), np.ravel(gen_traj([result.x[0],result.x[1], result.x[2], result.x[3]])[1,:]), 'b')
ax[1].set_ylabel("state 2")
ax[1].set_xlabel("time(s)")
ax[1].legend(["data", "model"])
plt.show()

print("Actual A matrix components", A_components)
print("Obtained A matrix components", result.x[0], result.x[1], result.x[2], result.x[3])