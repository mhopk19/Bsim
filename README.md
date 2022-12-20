# Introduction
In this project, Bayesian inference methods are compared with variational inference methods in a battery modeling scenario. The two models that are compared are the Extended Kalman Filter (Bayesian) and a Variational Autoencoder RNN (VRNN).

![ Alt text](vrnn_learning_soc_voltage_prediction_gif.gif) / ! [](vrnn_learning_soc_voltage_prediction_gif.gif)


# Instructions
To run the associated experiments run *estimator_tests.py*
The variable `using_EKF` will determine whether the EKF or VRNN are used


1. `vrnn_model.py` defines the VRNN model the model state dictionaries can be found in the "saves" folder
2. `vrnn_train.py` trains the VRNN model
3. `make_vrnn_data.py` makes batch data for training the VRNN. This data is saved to `vrnn_train_data.npy`
4. `ekf.py` stores the EKF
5. `battery_core.py` defines the battery object used for simulations