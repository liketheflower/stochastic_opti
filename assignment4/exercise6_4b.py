# this code is used to resolve the exercise 6.4 for the stochastic optimization class of the Graduate Center of CUNY 2017 Fall
# by jimmy shen Nov 30, 2017
import numpy as np
import matplotlib.pyplot as plt
import math




def get_x_theta(theta, u):
    return -50*theta*math.log(1-u)

def ipa(iteration = 100,theta_initial=5.0,learning_rate=0.0001 , plot=False):
    # generate uniform random numbers
    u = np.random.rand(iteration)
    theta = np.zeros((iteration,),dtype=np.float)
    # follow the Figure 6.2, the initial theta value is set to 5.0
    theta[0]= theta_initial
    for i in range(iteration-1):
        x_theta= get_x_theta(theta[i],u[i])
        d_h_x_theta =1/(1+x_theta)**0.5-x_theta/(2*(1+x_theta)**1.5)
        theta[i+1] = theta[i]- learning_rate*(1-(x_theta/theta[i])*d_h_x_theta)
    plt.scatter(list(range(iteration)), theta, s=2)
    plt.title(r'$\theta_n$'+' V.S. iteration, with '+r'$\epsilon = 0.2 $')
    plt.xlabel('iteration')
    plt.ylabel(r'$\theta$')
    plt.show()
    plt.close()

if __name__ == '__main__':
   iteration = 3000
   thetas =[5]
   learning_rate = 0.2
   for theta in thetas:
       ipa(iteration, theta, learning_rate)

