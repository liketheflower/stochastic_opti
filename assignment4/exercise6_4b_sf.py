# this code is used to resolve the exercise 6.4 for the stochastic optimization class of the Graduate Center of CUNY 2017 Fall
# by jimmy shen Nov 30, 2017
import numpy as np
import matplotlib.pyplot as plt
import math




def get_x_theta(theta, u):
    return -50*theta*math.log(1-u)

def get_h_x(x):
    return x/(x+1)**0.5

def get_gradient_ipa(x_theta , d_h_x_theta, theta):
    return (x_theta/theta)*d_h_x_theta
 
def get_gradient_sf(x_theta , h_x_theta, theta):
    return (1/theta**2)*((x_theta/theta)-1)

def get_gradient_mvd( h_x_theta,h_x_y_theta, theta):
    return (1/theta)*(h_x_y_theta-h_x_theta)
def sto_apro(method,iteration = 1000,theta_initial=5.0,learning_rate=0.0001 , plot=False):
    # generate uniform random numbers
    # u = np.random.rand(iteration)
    # available random number files:random_10000.npy  random_1000.npy  random_100.npy  random_2000.npy  random_5000.npy
    u = np.load('./random_number/random_'+str(iteration)+'.npy')
    # used for mvd
    u_y = np.load('./random_number/y_random_'+str(iteration)+'.npy')

    theta = np.zeros((iteration,),dtype=np.float)
    # follow the Figure 6.2, the initial theta value is set to 5.0
    theta[0]= theta_initial
    for i in range(iteration-1):
        x_theta= get_x_theta(theta[i],u[i])
        y_theta= get_x_theta(theta[i],u_y[i])
        d_h_x_theta =1/(1+x_theta)**0.5-x_theta/(2*(1+x_theta)**1.5)
        h_x_theta = get_h_x(x_theta)
        h_x_y_theta = get_h_x(x_theta+y_theta)
        #ipa
        # theta[i+1] = theta[i]- learning_rate*(1-(x_theta/theta[i])*d_h_x_theta)
        # ipa
        if method == 'ipa':
            estimated_gradient = get_gradient_ipa(x_theta,d_h_x_theta,theta[i])
        # sf
        if method == 'sf':
            estimated_gradient = get_gradient_sf(x_theta,h_x_theta,theta[i])
        if method == 'mvd':
            estimated_gradient = get_gradient_mvd(h_x_theta,h_x_y_theta, theta[i])
             
        theta[i+1] = theta[i]- learning_rate*(1-estimated_gradient)
    plt.scatter(list(range(iteration)), theta, s=2)
    plt.title(r'$\theta_n$'+' V.S. iteration, with '+r'$\epsilon = 0.2 $')
    plt.xlabel('iteration')
    plt.ylabel(r'$\theta$')
    plt.show()
    plt.close()
    return theta

if __name__ == '__main__':
   iteration = 10000
   theta_init =5.0
   learning_rate = 0.01
   theta_approx = sto_apro('ipa',iteration, theta_init, learning_rate)
   theta_approx = sto_apro('sf' ,iteration, theta_init, learning_rate)
   theta_approx = sto_apro('mvd',iteration, theta_init, learning_rate)
