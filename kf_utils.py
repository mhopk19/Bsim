import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciopt

def poopanya_VOC(soc):
    # open circuit voltage from interpolation function
    tabley = np.array([2.81, 3.23, 3.45, 3.56, 3.65, 3.76, 3.84, 3.91, 4.08, 4.12, 4.2])
    ocv = np.interp(soc, np.array([0,10,20,30,40,50,60,70,80,90,100]), tabley)
    return ocv


class Polynomial:
    def __init__(self, coeffs):
        self._coeffs = coeffs
        self._deg = len(coeffs) - 1

    def __call__(self, x):
        value = 0
        for i, coeff in enumerate(self._coeffs):
            value += coeff * x ** i
        return value

    @property
    def deriv(self):
        d_coeffs = [0]*self._deg
        for i in range(self._deg):
            d_coeffs[i] = (i+1)*self._coeffs[i+1]
        return Polynomial(d_coeffs)


if __name__ == '__main__':
    iterations = 0
    def draw_SOC_func(polynomial, color = 'r', input_scale = 1):
        y = []
        for i in range(100):
            xx = i/100
            y.append(polynomial(xx * input_scale))
        plt.plot(range(len(y)),y,color)
        
    def optimize_SOC_func():
        global iterations
        def loss(params):
            global iterations
            iterations = iterations + 1
            polynomial = Polynomial(params)
            y = []
            for i in range(100):
                xx = i/100
                y.append((polynomial(xx) - poopanya_VOC(xx*100))**2)
            cost = np.mean(np.array(y))
            plt.cla()
            draw_SOC_func(polynomial, color = 'r')
            draw_SOC_func(poopanya_VOC, color = 'b', input_scale = 100)
            plt.title("SOC polynomial opt.\n iter. {} cur parameters: {} \n loss: {}".format(iterations, params, cost))
            plt.ylabel("VOC")
            plt.xlabel("SOC")
            plt.legend(["polynomial", "measured"])
            if (iterations == 1):
                plt.pause(60)
            else:
                plt.pause(0.01)
            return cost
        result = sciopt.least_squares(loss,(3.1400, 3.9905, -14.2391, 24.4140, -13.5688, -4.0621, 4.5056),jac='3-point')
        return result
    
    my_poly = Polynomial([0])
    my_poly_deriv = my_poly.deriv
    print(my_poly._coeffs)
    print(my_poly_deriv._coeffs)
    print("result : ", my_poly(1))
    print("result : ", my_poly_deriv(1))
    my_poly = Polynomial([1,2,3,4])
    my_poly_deriv = my_poly.deriv
    print(my_poly._coeffs)
    print(my_poly_deriv._coeffs)
    print("result : ", my_poly(1))
    print("result : ", my_poly_deriv(1))
    
    ekf_poly = Polynomial([3.1400, 3.9905, -14.2391, 24.4140, -13.5688, -4.0621, 4.5056])
    draw_SOC_func(ekf_poly)
    plt.show()
    draw_SOC_func(poopanya_VOC, input_scale = 100)
    plt.show()
    result = optimize_SOC_func()
    print("optimized parameters", result)
    plt.cla()
    draw_SOC_func(Polynomial(list(result.x)))
    draw_SOC_func(poopanya_VOC, color = 'b', input_scale = 100)
    plt.title("Optimized OCV polynomial")
    plt.ylabel("VOC")
    plt.xlabel("SOC")
    plt.legend(["polynomial", "measured"])
    plt.show()
    