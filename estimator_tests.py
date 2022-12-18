from ekf import ExtendedKalmanFilter as EKF
import kf_utils as utils
import battery_core as bat
import numpy as np
import math
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # time period
    timestep = 0.1
    timesteps = 300
    inference_period = 200
    prediction_period = 100


    # create data
    load = True
    save = False
    using_EKF = True
    
    if (load):
        data = np.load("bat_tests.npy")
    else:    
        data = bat.create_data(batch_timesteps = timesteps, total_time = 30000, dt = 0.1)
    
    if (save):
        np.save("bat_tests", data)
        
    num_batches = data.shape[0]
    
    proxy_battery = bat.battery18650()
    
    
    if (using_EKF):
        Kf = EKF()
    else:
        pass
    

    SOC_estim_avg = np.zeros((inference_period))
    voltage_estim_avg = np.zeros((prediction_period))
    
    for batch in range(int(num_batches)):
        time         = [0]
        true_SoC = [data[batch][2][0]]
        estim_SoC = [Kf.x[0,0]]
        estim_SoC_error = [0]
        estim_voltage = [0]
        true_voltage = [data[batch][1][0]]
        mes_voltage = [data[batch][1][0] + np.random.normal(0,0.1,1)[0]]
        estim_voltage_error = [0]
        current = [data[batch][0][0]]
        
        """
        update these based on data ^^
        true_SoC     = [battery_simulation.state_of_charge]
        estim_SoC    = [Kf.x[0,0]]
        true_voltage = [battery_simulation.voltage]
        mes_voltage  = [battery_simulation.voltage + np.random.normal(0,0.1,1)[0]]
        current      = [battery_simulation.current]
        """
    
        # estimation period
        for i in range(inference_period):
            actual_current = data[batch][0][i]
            
            time.append(time[-1] + timestep)
            current.append(actual_current)
            mes_voltage.append(data[batch][1][i] + np.random.normal(0,0.1,1)[0])
            
            Kf.predict(u = actual_current)
            # MIGHT NEED TO FIX THE INPUT HERE
            Kf.update(mes_voltage[-1] + Kf.Rs * actual_current)
            
            true_SoC.append(data[batch][2][i])
            estim_SoC.append(Kf.x[2,0])
            estim_SoC_error.append(math.sqrt((100*(true_SoC[-1] - estim_SoC[-1]))**2))
            
            estim_voltage.append(Kf.OCV_func(Kf.x[2,0]) - Kf.x[0,0] - Kf.x[1,0] - Kf.Rs*actual_current)
            print("estimates", Kf.x)
        
        # update average soc estimation value
        SOC_estim_avg = SOC_estim_avg + np.array(estim_SoC_error)[1:] / num_batches
        
        # prediction period
        # calibrate battery
        proxy_battery.x[:-1] = np.array(Kf.x).squeeze()
        
        for i in range(inference_period, inference_period + prediction_period):
            actual_current = data[batch][0][i]
            proxy_battery.step(actual_current)
    
            time.append(time[-1] + timestep)
            current.append(actual_current)
            mes_voltage.append(data[batch][1][i] + np.random.normal(0,0.1,1)[0])
            
            # estimated voltage is the proxy batteries voltage
            estim_voltage.append(proxy_battery.x[3])
            
            estim_voltage_error.append(math.sqrt(((mes_voltage[-1] - estim_voltage[-1]))**2))
          
        plt.cla()  
        plt.plot(range(inference_period, inference_period + prediction_period), mes_voltage[inference_period+1:], 'r')
        plt.plot(range(inference_period, inference_period + prediction_period), estim_voltage[inference_period+1:], 'r--')
        plt.ylabel("V")
        plt.xlabel("time step {}s".format(timestep))
        plt.legend(["measured voltage", "pred. voltage"])
        plt.title("Voltage Tracking")
        plt.pause(0.1)
        
        # update average soc estimation value
        voltage_estim_avg = voltage_estim_avg + np.array(estim_voltage_error)[1:] / num_batches  
    
    plt.cla()
    plt.plot(range(inference_period), SOC_estim_avg, 'r')
    plt.ylabel("Squared Error (V)")
    plt.xlabel("time step {}s".format(timestep))
    plt.title("EKF Mean Squared SOC Error")
    plt.savefig("./results/EKF_average_estimation_error")

    plt.cla()
    plt.plot(range(inference_period, inference_period + prediction_period), voltage_estim_avg, 'r')
    plt.ylabel("Squared Error (%)")
    plt.xlabel("time step {}s".format(timestep))
    plt.title("EKF Mean Squared Voltage Error")
    plt.savefig("./results/EKF_average_voltage_error")    
    
    
    
    
    
    
    