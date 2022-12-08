import numpy as np
import math
import matplotlib.pyplot as plt

# x_{n+1} = x_{n} + (Df(x_n))^(-1) f(x_n)

# simulating     f_1 = xy + x^2 - y^3 - 1
#                f_2 = x + 2y - xy^2 - 2

def f(x,y):
    f1 = x*y + x*x - y*y*y - 1
    f2 = x + 2*y - x*y*y - 2
    
    return np.array([f1, f2])

def Df(x,y):
    D = np.array([[y+2*x, x-3*y*y],
                  [1-y*y, 2-2*x*y]])
    return D
    

def inv_Df(x,y):
    inv = 1/((y+2*x)*(2-2*x*y) - (x-3*y*y)*(1-y*y))
    return inv*Df(x,y)

t_final = 0.57

t = 0
tol = 0.001
dt = 0.01
error = 0
x = np.array([1,1])
x_hist = np.array([0,0])

while(t<t_final):
    while (error > tol):
        x_new = x - inv_Df(x[0],x[1])*f(x[0],x[1])
        error = f(x_new[0],x_new[1])
        x = x_new
        
    print("error", error)
    x = x + dt*Df(x[0],x[1]) @ x
    x_hist = np.vstack((x_hist,x))
    t = t + dt
    
for i,x in enumerate(x_hist):
    print("i:{} x:{}".format(i*dt,x))

plt.plot(range(x_hist.shape[0]), x_hist[:,0],'b')
plt.plot(range(x_hist.shape[0]), x_hist[:,1],'r')
plt.show()