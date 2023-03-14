import matplotlib.pyplot as plt
import numpy as np

soc_values = np.array([0,10,20,30,40,50,60,70,80,90, 100])
Rs_table = np.array([56, 56, 45, 42, 43, 42, 43, 44, 42, 50, 50]) / 1000
Rts_table = np.array([13, 13, 20, 12, 15, 16, 17, 16, 17, 19, 19]) / 1000
Rtl_table = np.array([10, 10, 9, 9, 18, 13, 7, 10, 100, 10, 10]) / 1000
Cts_table = np.array([11, 11, 0.45, 0.5, 3, 7, 4, 3, 1, 3, 3]) * 1000
Ctl_table = np.array([5, 5, 79, 20, 19, 1, 42, 183, 5, 100, 100]) * 1000


fig1, ax1 = plt.subplots(3,1)
fig2, ax2 = plt.subplots(2,1)

# plotting resistor values
ax1[0].plot(soc_values, Rs_table, 'r')
ax1[1].plot(soc_values, Rts_table, 'r')
ax1[2].plot(soc_values, Rtl_table, 'r')
ax1[0].set_ylabel(r'$\Omega$ ($R_{int}$)')
ax1[1].set_ylabel(r'$\Omega$ ($R_{ts}$)')
ax1[2].set_ylabel(r'$\Omega$ ($R_{tl}$)')
ax1[2].set_xlabel(r'SOC (%)')
ax1[0].set_title("ECF Parameters: Resistors")

# plotting capacitor values
ax2[0].plot(soc_values, Cts_table, 'b')
ax2[1].plot(soc_values, Ctl_table, 'b')
ax2[0].set_ylabel(r'Farads ($C_{ts}$)')
ax2[1].set_ylabel(r'Farads ($C_{tl}$)')
ax2[1].set_xlabel(r'SOC (%)')
ax2[0].set_title("ECF Parameters: Capacitors")

plt.show()





