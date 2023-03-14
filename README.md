# Introduction
In this project, Bayesian inference methods are compared with variational inference methods in a battery modeling scenario. The two models that are compared are the Extended Kalman Filter (Bayesian) and a Variational Autoencoder RNN (VRNN).

![ Alt text](vrnn_learning_soc_voltage_prediction_gif.gif) / ! [](vrnn_learning_soc_voltage_prediction_gif.gif)


# Instructions
Run *estimator_tests.py* to perform the associated experiments.
The variable `using_EKF` will determine whether the EKF or VRNN is used


1. `vrnn_model.py` defines the VRNN model the model state dictionaries can be found in the "saves" folder
2. `vrnn_train.py` trains the VRNN model
3. `make_vrnn_data.py` makes batch data for training the VRNN. This data is saved to `vrnn_train_data.npy`
4. `ekf.py` stores the EKF
5. `battery_core.py` defines the battery object used for simulations


# Testing
Modules can be tested using the 'pytest' command within the 'tests' folder. All test files should be contained in files with "test" prefix or "test.py" suffix.
All test methods should have a "test" prefix


# Modeling



# Battery Pack Simulation Method 

Referenced from Plett 2015 lecture notes "ECE5720: Battery Management and Control"


## Series Connections

<!--- these styles work
```math
\begin{align}y&=5\\x&=8\end{align}
```
-->
```math
\begin{align}
y&=5\\
x&=8\\
&\frac{5}{6}
\end{align}
```

$$i_k$$ the applied current is given
```math
\begin{align}
v_{pack}(t) &= (\sum_{k=1}^N_s v_{cell,k}(t) ) - N_{cells}R_{interconnect}i(t)
\end{align}
```

## Parallel Connections
$$v_{j,k}$$ fixed voltage for branch j at time k
```math
\begin{align}
v_{pack}(t) &= \frac{\sum_{j=1}^N_{p} \frac{v_{j,k}(t)}{R_{0,j}} - i_k}{\sum_{j=1}^N_{p} \frac{1}{R_{0,j}} }
\end{align}
```
From $$v_k$$ we can find the individual branch currents
```math
i_{j,k} = \frac{v_{j,k} - v_k}{R_{0,j}}
```
