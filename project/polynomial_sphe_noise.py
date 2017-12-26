# this code is used to generate a simle plot of : y=x**2+(1-x)**2
# by jimmy shen 
# Dec 12, 2017
import math
import numpy as np
import matplotlib.pyplot as plt
class Simple(object):
    def __init__(self, step=0.01, lower=-5, upper =5):
        print "this is a simple example"
        if step>1.0 or step <0.0:
            print "error: step has to be greater than 0 and less than 1.0"
        self.x = [i*step for i in range(int(lower/step),int(upper/step))]
   
    def plot_(self, y,title,file_name,x_lab='x', y_lab=r'$J(x)$'):
        y_ = [-_ for _ in y]
        plt.plot(self.x, y_)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.title(title)
        file_name = './result/'+file_name
        plt.savefig(file_name, format='eps', dpi=1000)
        plt.show()
        plt.close() 

def get_d_j(x):
    return 8*math.cos(x)**7*math.sin(x)-8*math.cos(x)**3*math.sin(x)+2*math.cos(x)*math.sin(x)+np.random.normal(0,1)


def simulation(iteration, learning_rate, alpha_n):
    res=[]
    estimated_theta=[]
   # estimate_x=[]
    for i, a in enumerate(random_):
#        print 'i,a',i,a
        tem = [a*math.pi/2.0]
        for i in range(iteration):
            tem.append(tem[i-1]-learning_rate*get_d_j(tem[i-1]))
        res+=tem
        estimated_theta.append(tem[-1])
    return res, estimated_theta

def plot_new(x, y,title,file_name,x_lab='x', y_lab=r'$J(x)$'):
       # y_ = [-_ for _ in y]
        plt.plot(x, y)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.title(title)
        file_name = './result/'+file_name
        plt.savefig(file_name, format='eps', dpi=1000)
        plt.show()
        plt.close()




if __name__=='__main__':
    import time
    start_time = time.time()
    iteration = 500
    learning_rate = 0.01
    repeat = 5
    random_a = np.load('random100.npy') 
    random_ = random_a[:repeat]
    res,estimated_theta = simulation(iteration, learning_rate, random_)
    import collections
    optimal_theta = collections.defaultdict(float)
    for x_ in estimated_theta:
        optimal_theta[x_] = -math.cos(x_)**8+2*math.cos(x_)**4-math.cos(x_)
    optimal_theta_ = sorted(optimal_theta, key=optimal_theta.get)
    print 'res', math.cos(optimal_theta_[0])**2
    print("--- %s seconds ---" % (time.time() - start_time))
   # print 'res',min[math.cos(_) for _ in estimated_theta]
    plot_new(list(range(len(res))),res,"Resuls of the spherical coordinates parameterization",'polynomial_est_sphe_noise.eps',r'$n$',r'$\theta_n$')
    #plt.show()

