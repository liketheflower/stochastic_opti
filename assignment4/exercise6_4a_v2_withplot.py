
#this code is used to resolve the exercise 6.4 for the stochastic optimization class of the Graduate Center of CUNY 2017 Fall
# by jimmy shen Nov 30, 2017
import numpy as np
import matplotlib.pyplot as plt
import math
#import scipy as sp
#import scipy.stats as st
#import numpy as np, scipy.stats as st
from scipy import stats
import time
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
 
def plot_result(method,a,title, file_name='nothing'):
     count, bins, ignored = plt.hist(a, 100, normed=True, lw=3, fc=(0,0,1,0.5))
        #plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
 #linewidth=2, color='r')
     plt.title(title)
     plt.savefig(method.upper()+'hist.eps', format='eps', dpi=1000)
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
    return (1/float(theta))*h_x_theta*((x_theta/float(50*theta))-1)

def get_gradient_mvd( h_x_theta,h_x_y_theta, theta):
   # print "h_x_theta,h_x_y_theta, theta,(1/theta)*(h_x_y_theta-h_x_theta)"
   # print h_x_theta,h_x_y_theta, theta,(1/float(theta))*(h_x_y_theta-h_x_theta)
    return (1/float(theta))*(h_x_y_theta-h_x_theta)


def estimate_gradients(method,u,u_y,theta,plot=True):
    estimated_gradient = np.zeros(u.shape,dtype=np.float)
    for i in range(u.shape[0]):
        x_theta = get_x_theta(theta, u[i])
        d_h_x_theta =1/(1+x_theta)**0.5-x_theta/(2*(1+x_theta)**1.5)
        estimated_gradient[i] = (x_theta/theta)*d_h_x_theta
        
        x_theta= get_x_theta(theta,u[i])
        y_theta= get_x_theta(theta,u_y[i])
        d_h_x_theta =1/(1+x_theta)**0.5-x_theta/(2*(1+x_theta)**1.5)
        h_x_theta = get_h_x(x_theta)
        h_x_y_theta = get_h_x(x_theta+y_theta)
        #ipa
        # theta[i+1] = theta[i]- learning_rate*(1-(x_theta/theta[i])*d_h_x_theta)
        # ipa
        if method == 'ipa':
            estimated_gradient[i] = get_gradient_ipa(x_theta,d_h_x_theta,theta)
        # sf
        if method == 'sf':
            estimated_gradient[i] = get_gradient_sf(x_theta,h_x_theta,theta)
        if method == 'mvd':
            estimated_gradient[i] = get_gradient_mvd(h_x_theta,h_x_y_theta, theta)

    if plot and theta == 10:
        plot_result(method, estimated_gradient,'Hist of the estimated gradient based'+method.upper()+'('+r'$\theta=$'+str(theta) +', 10000 experiments)')

    return estimated_gradient


def estimate_and_print(method,u,u_y,theta):
    start_time = time.time()
    estimated_gradient =  estimate_gradients(method,u,u_y,theta)
    mean,std, (confident_interval_left, confident_interval_right) =mean_confidence_interval(estimated_gradient)
      #print theta, np.mean(estimated_gradient), np.std(estimated_gradient)
   # print method, estimated_gradient.shape
    print method.upper(), theta, mean,std, confident_interval_left, confident_interval_right,time.time() - start_time
if __name__ == '__main__':
# available random number files:random_10000.npy  random_1000.npy  random_100.npy  random_2000.npy  random_5000.npy 
   iteration = 10000
   u = np.load('./random_number/random_'+str(iteration)+'.npy')
   u_y = np.load('./random_number/y_random_'+str(iteration)+'.npy')
   thetas =[6,8,10,12]
  # learning_rate = 0.2
#   test = np.random.normal(0,1,100000)
 #  e,(f,g)= mean_confidence_interval(test)
   print r'$\theta$','average','std','cnfidence interval(95%)','time'
  # print "--- %s seconds ---" % (time.time() - start_time) ,method,"theta=", theta

   for theta in thetas:
      estimate_and_print('ipa',u,u_y, theta)
      estimate_and_print('sf',u,u_y, theta)
      estimate_and_print('mvd',u,u_y, theta)




