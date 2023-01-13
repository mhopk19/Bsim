import numpy as np
import argparse
import matplotlib.pyplot as plt

"""
index 0 motor voltage
index 1 battery voltage
index 2 motor current
index 3 rotations
index 4 time
"""
parser = argparse.ArgumentParser()

# save file
parser.add_argument("-f",
                    "--file",
                    type=str,
                    default="9v_motor_sys_id.npy",
                    help="whether to save data and file name")

args = parser.parse_args()

data = np.load(args.file)
print("data shape", data.shape)

fig, ax = plt.subplots(2,2)# sharex='col', sharey='row')
plt.suptitle("PM Motor System Dynamics")

ax[0,0].plot(range(len(data[:,0])), data[:,0],"b")
ax[0,0].set_ylabel(r'$V_{mtr}(t)$')

ax[0,1].plot(range(len(data[:,1])), data[:,1],"r")
ax[0,1].set_ylabel(r'$V_{bat}(t)$')

ax[1,0].plot(range(len(data[:,2])), data[:,2],"k")
ax[1,0].set_ylabel(r'$i_a(t)$')
ax[1,0].set_xlabel("Time (s)")

ax[1,1].plot(range(len(data[:,3])), data[:,3],"y")
ax[1,1].set_ylabel(r'$\omega(t)$')
ax[1,1].set_xlabel("Time (s)")
#plt.plot(range(len(data[:,4])), data[:,4],"k")

plt.show()
