# this code is used to resolve the exercise 6.4 for the stochastic optimization class of the Graduate Center of CUNY 2017 Fall
# by jimmy shen Nov 30, 2017
import numpy as np
import matplotlib.pyplot as plt
import math
#import scipy as sp
#import scipy.stats as st
#import numpy as np, scipy.stats as st
from scipy import stats
def mean_confidence_interval(a, confidence=0.68):
    #a = 1.0*np.array(data)
   # n = a.shape[0]
   # m, se = np.mean(a), scipy.stats.sem(a)
   # h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    mean, sigma = np.mean(a), np.std(a)
    interval = stats.norm.interval(confidence, loc=mean, scale=sigma)
   # interval=st.t.interval(confidence, a.shape[0]-1, loc=np.mean(a), scale=st.sem(a))
   # print interval
    return mean,sigma, interval


def get_x_theta(theta, u):
    return -50*theta*math.log(1-u)

def plot_result(a,title, file_name='nothing'):
     count, bins, ignored = plt.hist(a, 100, normed=True, lw=3, fc=(0,0,1,0.5))
        #plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
 #linewidth=2, color='r')
     plt.title(title)
     plt.show()
     plt.close()

def ipa(u,theta,plot=True):
    estimated_gradient = np.zeros(u.shape,dtype=np.float)
    for i in range(u.shape[0]):
        x_theta = get_x_theta(theta, u[i])
        d_h_x_theta =1/(1+x_theta)**0.5-x_theta/(2*(1+x_theta)**1.5)
        estimated_gradient[i] = (x_theta/theta)*d_h_x_theta
    if plot and theta == 10:
        plot_result(estimated_gradient,'Hist of the estimated gradient based on IPA('+r'$\theta=$'+str(theta) +', 10000 experiments)')

    return estimated_gradient
"""
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
"""



if __name__ == '__main__':
# available random number files:random_10000.npy  random_1000.npy  random_100.npy  random_2000.npy  random_5000.npy 
   iteration = 10000
   u = np.load('./random_number/random_'+str(iteration)+'.npy')
   thetas =[6,8, 10,12]
  # learning_rate = 0.2
#   test = np.random.normal(0,1,100000)
 #  e,(f,g)= mean_confidence_interval(test)
   for theta in thetas:
      estimated_gradient =  ipa(u, theta)
      mean,std, (confident_interval_left, confident_interval_right) =mean_confidence_interval(estimated_gradient)
      #print theta, np.mean(estimated_gradient), np.std(estimated_gradient)
      print r'$\theta$', theta,'average', mean,'std',std,'cnfidence interval(95%):[', confident_interval_left,',', confident_interval_right,']'
      




