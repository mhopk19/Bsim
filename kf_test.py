from ekf import ExtendedKalmanFilter as EKF
import kf_utils as utils
import battery_core as bat
import numpy as np
import math as m
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # time period
    timestep = 0.1
    timesteps = 200

    data = bat.create_data(batch_timesteps = timesteps, total_time = 20000, dt = 0.1)
    print("data shape", data.shape)
    
    #get configured EKF
    Kf = EKF()
    
    # batch we are zeroing in on
    batch = 12

    time         = [0]
    true_SoC = [data[batch][2][0]]
    estim_SoC = [Kf.x[0,0]]
    estim_voltage = [0]
    true_voltage = [data[batch][1][0]]
    mes_voltage = [data[batch][1][0] + np.random.normal(0,0.1,1)[0]]
    current = [data[batch][0][0]]
    
    """
    update these based on data ^^
    true_SoC     = [battery_simulation.state_of_charge]
    estim_SoC    = [Kf.x[0,0]]
    true_voltage = [battery_simulation.voltage]
    mes_voltage  = [battery_simulation.voltage + np.random.normal(0,0.1,1)[0]]
    current      = [battery_simulation.current]
    """

    for i in range(timesteps):
        actual_current = data[batch][0][i]
        
        time.append(time[-1] + timestep)
        current.append(actual_current)
        mes_voltage.append(data[batch][1][i] + np.random.normal(0,0.1,1)[0])
        
        Kf.predict(u = actual_current)
        # MIGHT NEED TO FIX THE INPUT HERE
        Kf.update(mes_voltage[-1] + Kf.Rs * actual_current)
        
        true_SoC.append(data[batch][2][i])
        estim_SoC.append(Kf.x[2,0])
        estim_voltage.append(Kf.OCV_func(Kf.x[2,0]) - Kf.x[0,0] - Kf.x[1,0] - Kf.Rs*actual_current)
        print("estimates", Kf.x)
        
    plt.plot(range(timesteps+1), true_SoC,'g')
    plt.plot(range(timesteps+1), estim_SoC,'g--')
    plt.plot(range(timesteps+1), current,'k')
    plt.plot(range(timesteps+1), estim_voltage,'r')
    plt.show()