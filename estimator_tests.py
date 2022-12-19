from ekf import ExtendedKalmanFilter as EKF
import kf_utils as utils
import battery_core as bat
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from vrnn_model import VRNN

if __name__ == '__main__':
    # time period
    timestep = 0.1
    timesteps = 300
    inference_period = 200
    prediction_period = 100


    # create data
    load = True
    save = False
    using_EKF = False
    
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
        # create vrnn
        x_dim = 21
        h_dim = 100
        z_dim = 16
        n_layers =  1
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        state_dict = torch.load('saves/vrnn_state_dict_41.pth')
        vrnn_model = VRNN(x_dim, h_dim, z_dim, n_layers)
        vrnn_model.load_state_dict(state_dict)
        vrnn_model.to(device)
    

    SOC_estim_avg = np.zeros((inference_period))
    voltage_estim_avg = np.zeros((prediction_period))
    bound_fault_avg = np.zeros((prediction_period))
    
    for batch in range(int(num_batches)):
        time         = [0]
        true_SoC = [data[batch][2][0]]
        
        if (using_EKF == True):
            estim_SoC = [Kf.x[0,0]]
        else:
            estim_SoC = [0.5]
            
            
        estim_SoC_error = [0]
        estim_voltage = [0]
        true_voltage = [data[batch][1][0]]
        mes_voltage = [data[batch][1][0] + np.random.normal(0,0.1,1)[0]]
        estim_voltage_error = [0]
        current = [data[batch][0][0]]
        bound_fault = [0]
        
        # make Kalman filter soc guess near true soc
        if (using_EKF == True):
            Kf.x[2,0] = max(0,min(1, data[batch][2][0] + np.random.normal() * 1/3))

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
            
            if (using_EKF == True):
                Kf.predict(u = actual_current)
                # MIGHT NEED TO FIX THE INPUT HERE
                Kf.update(mes_voltage[-1] + Kf.Rs * actual_current)
                estim_SoC.append(Kf.x[2,0])
                estim_voltage.append(Kf.OCV_func(Kf.x[2,0]) - Kf.x[0,0] - Kf.x[1,0] - Kf.Rs*actual_current)
            else:
                if (i < 10):
                    estim_SoC.append(0.5)
                    estim_voltage.append(0)
                else:
                    inputs = torch.tensor(current[i-10:i])
                    voltages = torch.tensor(mes_voltage[i-10:i])
                    # 0 as a place holder for soc
                    vrnn_input = torch.unsqueeze(torch.unsqueeze(torch.cat([inputs,voltages,torch.tensor([0])]),0),0).float()
                    print("vrnn input shape", vrnn_input)
                    estimates = vrnn_model.predict(vrnn_input, reset = (i==10))
                    print("estimates", estimates)
                    estim_SoC.append(estimates[0][0][0][1])
                    estim_voltage.append(estimates[0][0][0][0])
            
                
            
            true_SoC.append(data[batch][2][i])
            estim_SoC_error.append(math.sqrt((100*(true_SoC[-1] - estim_SoC[-1]))**2))
            
        
        # update average soc estimation value
        SOC_estim_avg = SOC_estim_avg + np.array(estim_SoC_error)[1:] / num_batches
        
        # prediction period
        # calibrate battery
        if (using_EKF == True):
            proxy_battery.x[:-1] = np.array(Kf.x).squeeze()
        else:
            estim_bounds = []
        
        for i in range(inference_period, inference_period + prediction_period):
            time.append(time[-1] + timestep)
            actual_current = data[batch][0][i]
            current.append(actual_current)
            
            if (using_EKF == True):
                proxy_battery.step(actual_current)
                # estimated voltage is the proxy batteries voltage
                estim_voltage.append(proxy_battery.x[3])
            else:
                inputs = torch.tensor(current[i-10:i])
                voltages = torch.tensor(mes_voltage[i-10:i])
                # 0 as a place holder for soc
                vrnn_input = torch.unsqueeze(torch.unsqueeze(torch.cat([inputs,voltages,torch.tensor([0])]),0),0).float()
                estimates = vrnn_model.predict(vrnn_input, reset = False)
                estim_voltage.append(estimates[0][0][0][0])
                estim_bounds.append(estimates[1][0][0][0])
                
            cur_voltage = data[batch][1][i] + np.random.normal(0,0.1,1)[0]
            mes_voltage.append(cur_voltage)
            
            if (using_EKF == False):
                if (abs(estimates[0][0][0][0] - cur_voltage) > estimates[1][0][0][0]):
                    bound_fault.append(1)
                else:
                    bound_fault.append(0)
            
            estim_voltage_error.append(math.sqrt(((mes_voltage[-1] - estim_voltage[-1]))**2))
          
        plt.cla()  
        plt.plot(range(inference_period, inference_period + prediction_period), mes_voltage[inference_period+1:], 'r')
        plt.plot(range(inference_period, inference_period + prediction_period), estim_voltage[inference_period+1:], 'r*')
        plt.errorbar(range(inference_period, inference_period + prediction_period),estim_voltage[inference_period+1:],yerr = estim_bounds)
        plt.ylabel("V")
        plt.xlabel("time step {}s".format(timestep))
        plt.legend(["measured voltage", "pred. voltage"])
        plt.title("Voltage Tracking")
        plt.pause(0.1)
        
        # update average soc estimation value
        voltage_estim_avg = voltage_estim_avg + np.array(estim_voltage_error)[1:] / num_batches  
        bound_fault_avg = bound_fault_avg + np.array(bound_fault)[1:] 
    
    if (using_EKF == True):
        model_str = "EKF"
    else:
        model_str = "VRNN"
        # draw bound faults
        plt.cla()
        plt.plot(range(inference_period, inference_period + prediction_period), bound_fault_avg, 'r')
        plt.ylabel("Bound Faults (V)")
        plt.xlabel("time step {}s".format(timestep))
        plt.title("VRNN Bound Faults Stochastic Precision (avg. #)")
        plt.savefig("./results/VRNN_stochastic_bounds") 
    
    plt.cla()
    plt.plot(range(inference_period), SOC_estim_avg, 'r')
    plt.ylabel("Squared Error (%)")
    plt.xlabel("time step {}s".format(timestep))
    plt.title("{} Mean Squared SOC Error".format(model_str))
    plt.savefig("./results/{}_average_estimation_error".format(model_str))

    plt.cla()
    plt.plot(range(inference_period, inference_period + prediction_period), voltage_estim_avg, 'r')
    plt.ylabel("Squared Error (V)")
    plt.xlabel("time step {}s".format(timestep))
    plt.title("{} Mean Squared Voltage Error".format(model_str))
    plt.savefig("./results/{}_average_voltage_error".format(model_str))    

    
    print("number of batches", num_batches)
    
    
    
    
    