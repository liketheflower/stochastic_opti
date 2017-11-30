# this code is used to resolve the exercise 6.4 for the stochastic optimization class of the Graduate Center of CUNY 2017 Fall
# by jimmy shen Nov 30, 2017
import numpy as np
import matplotlib.pyplot as plt
import math


from exercise_4a import get_x_theta,get_h_x,get_gradient_ipa,get_gradient_sf,get_gradient_mvd
"""

def get_x_theta(theta, u):
    return -50*theta*math.log(1-u)

def plot_result(a,title, file_name='nothing'):
     count, bins, ignored = plt.hist(a, 100, normed=True, lw=3, fc=(0,0,1,0.5))
        #plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
 #linewidth=2, color='r')
     plt.title(title)
     plt.show()
     plt.close()

def get_h_x(x):
    return float(x)/(float(x)+1)**0.5

def get_gradient_ipa(x_theta , d_h_x_theta, theta):
#    print "x_theta , d_h_x_theta, theta,(x_theta/theta)*d_h_x_theta"
 #   print x_theta , d_h_x_theta, theta,(x_theta/float(theta))*d_h_x_theta
    return (x_theta/float(theta))*d_h_x_theta

def get_gradient_sf(x_theta , h_x_theta, theta):
  #  print "x_theta , h_x_theta, theta, h_x_theta*(x_theta/theta-1),(1/float(theta))*h_x_theta*(x_theta/float(theta)-1)"
   # print x_theta , h_x_theta, theta,h_x_theta*(x_theta/float(theta)-1)*theta, (1/float(theta))*h_x_theta*(x_theta/float(theta)-1)
    return (1/float(theta))*h_x_theta*((x_theta/float(theta))-1)

def get_gradient_mvd( h_x_theta,h_x_y_theta, theta):
   # print "h_x_theta,h_x_y_theta, theta,(1/theta)*(h_x_y_theta-h_x_theta)"
   # print h_x_theta,h_x_y_theta, theta,(1/float(theta))*(h_x_y_theta-h_x_theta)
    return (1/float(theta))*(h_x_y_theta-h_x_theta)

"""







"""
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
"""
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
    plt.scatter(list(range(iteration)), theta, s=0.5)
    plt.title(method.upper()+" "+r'$\theta_n$'+' V.S. iteration, with '+r'$\epsilon =  $'+str(learning_rate))
    plt.xlabel('iteration')
    plt.ylabel(r'$\theta$')
    plt.savefig(method.upper()+'_sa.eps', format='eps', dpi=1000)
   # plt.show()
    plt.close()
    return theta

if __name__ == '__main__':
   iteration = 5000
   theta_init =5.0
   learning_rate = 0.06
   theta_approx = sto_apro('ipa',iteration, theta_init, learning_rate)
   theta_approx = sto_apro('sf' ,iteration, theta_init, learning_rate)
   theta_approx = sto_apro('mvd',iteration, theta_init, learning_rate)
