from numpy import zeros, eye
import kf_utils as utils
import numpy as np

# empirical Poopanya et. al found parameters
"""
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
"""

default_params = [0.1,0.2,0.4,0.5,0.5]

class ExtendedKalmanFilter(object):

    #def __init__(self):
    #    self.randles_circuit()
    
     
    def __init__(self, *args):
        self.time_step = 0.1
        self.std_dev = 0.1
        self.Q_tot = 1
        if (len(args) == 0):
            self.randles_circuit()
        else:
            x, F, B, P, Q, R = [*args]
            self.x = x
            self.F = F
            self.B = B
            self.P = P
            self.Q = Q
            self.R = R
            #self._Hx = Hx
            #self._HJacobian = HJacobian
            self.time_step = 0.1
            self.std_dev = 0.1
            self.Q_tot = 1
        
        
    def randles_circuit(self, *args):
        # initial state (SoC is intentionally set to a wrong value)
        # x = [[SoC], [RC voltage]]
        if (len(args) == 0):
            # parameters for 100% soc poopanya
            Rs, Rts, Cts, Rtl, Ctl = [56, 13, 11, 10, 5]
        else:
            Rs, Rts, Cts, Rtl, Ctl = [*args]
        self.Rs = Rs
        self.Rts = Rts
        self.Cts = Cts
        self.Rtl = Rtl
        self.Ctl = Ctl
        
        self.x = np.matrix([[0.5],\
                       [0.0],
                       [0.0]])
    
        exp_ts = (-self.time_step/(Cts*Rts))
        exp_tl = (-self.time_step/(Ctl*Rtl))
        
        # state transition model
        self.F = np.matrix([[exp_ts, 0, 0],\
                       [0, exp_tl, 0],
                       [0,     0,  0]])
    
        # control-input model
        self.B = np.matrix([[self.time_step/Cts],\
                       [self.time_step/Ctl],
                       [self.time_step/(self.Q_tot * 3200)]])
    
        # variance from std_dev
        var = self.std_dev ** 2
    
        # measurement noise
        self.R = var
    
        # state covariance
        self.P = np.matrix([[var, 0, 0],\
                       [0, var, 0],
                       [0,  0,  var]])
    
        # process noise covariance matrix
        self.Q = np.matrix([[var/50, 0,  0],\
                       [0, var/50,  0],
                       [0,     0,  var/50]])
    


    def G_prime_update(self):
        OCV_function = utils.Polynomial([2.87, 4.75, -14.4, 24.1, -13.72, -4.04, 4.67])  
        self.G_prime = np.matrix([[-1, - 1, OCV_function.deriv(self.x[0,0])]])
        return self.G_prime
    
    def G_update(self):
        OCV_function = utils.Polynomial([2.87, 4.75, -14.4, 24.1, -13.72, -4.04, 4.67])  
        self.G = OCV_function(self.x[0,0]) - self.x[1,0]
        return self.G
 
    


    def predict_horizon(self, tau, u_series):
        # tau is the time horizon
        #converged_P = np.array([[2.3e-05,2.85e-06],[2.85e-06,8.96e-06]])
        converged_P = np.array([[0.01,0.01],[0.01,0.01]])
        next_state = self._x
        next_covariance = converged_P 
        pred = np.zeros((tau))
        pred_cov = np.zeros((tau))
        for i in range(tau):
            print("i {} next state {} pred {}".format(i, next_state, pred))
            next_state = self._F * next_state + self._B * u_series[i]
            next_covariance = self._F @ next_covariance @ self._F.T
            pred[i] = next_state[0]
            print("next covariance", next_covariance)
            pred_cov[i] = next_covariance[0,0]
        return pred, pred_cov

    def update(self, z):

        P = self.P
        R = self.R
        x = self.x

        G_prime = self.G_prime_update()
        G = self.G_update()

        S = G_prime * P * G_prime.T + R
        K = P * G_prime.T * S.I
        self._K = K

        hx =  G
        y = np.subtract(z, hx)
        self._x = x + K * y

        KH = K * G_prime
        I_KH = np.identity((KH).shape[1]) - KH
        self._P = I_KH * P * I_KH.T + K * R * K.T

    def predict(self, u=0):
        self.x = self.F * self.x + self.B * u
        self.P = self.F * self.P * self.F.T + self.Q
