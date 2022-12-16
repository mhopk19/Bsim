from numpy import zeros, eye

import numpy as np


class ExtendedKalmanFilter(object):

    def __init__(self, x, F, B, P, Q, R, Hx, HJacobian):

        self._x = x
        self._F = F
        self._B = B
        self._P = P
        self._Q = Q
        self._R = R
        self._Hx = Hx
        self._HJacobian = HJacobian


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

        P = self._P
        R = self._R
        x = self._x

        H = self._HJacobian(x)

        S = H * P * H.T + R
        K = P * H.T * S.I
        self._K = K

        hx =  self._Hx(x)
        y = np.subtract(z, hx)
        self._x = x + K * y

        KH = K * H
        I_KH = np.identity((KH).shape[1]) - KH
        self._P = I_KH * P * I_KH.T + K * R * K.T

    def predict(self, u=0):
        self._x = self._F * self._x + self._B * u
        self._P = self._F * self._P * self._F.T + self._Q

    @property
    def x(self):
        return self._x
