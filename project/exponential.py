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

def get_d_j_a_n(x,a):
#    print 'x, a', x, a
    d_j =  (1+math.log(x+0.001))/((x+0.001)**2*math.sqrt(2*math.pi))*math.exp(-(math.log(x+0.001))**2/2.0)-0.01*math.exp(x)
#    print (1+math.log(x+0.001))/((x+0.001)**2*math.sqrt(2*math.pi))*math.exp(-math.log((x+0.001))**2/2.0)
 #   print 0.01*math.exp(x)
    if x <=1 and x>=0:
        return d_j
    else:
        return d_j+a*x


def simulation(iteration, learning_rate, alpha_n):
    res=[]
    smooth =10
   # estimate_x=[]
 
    x_init=10
    for i, a in enumerate(alpha_n):
#        print 'i,a',i,a
        tem = []
        for i in range(iteration):
            if i==0:
                tem.append(x_init)
            x_n =min(tem[i-1]-learning_rate*get_d_j_a_n(tem[i-1],a),50)
            tem.append(x_n)
        res+=tem
        average_ = sum(tem[-smooth:])/float(smooth)
        if 1000*average_>=0 and 1000*average_<=1000: 
            return res , average_
    return res, average_

def plot_new(x, y,title,file_name,x_lab='x', y_lab=r'$J(x)$'):
       # y_ = [-_ for _ in y]
        plt.plot(x, y)
        plt.ylim(0,51) 
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
    penalty_max_n = 100 
    alpha_n = [math.exp(i) for i in range(penalty_max_n)]
    res,estimated_x = simulation(iteration, learning_rate, alpha_n)
    print 'res', res[-1],estimated_x
    print("--- %s seconds ---" % (time.time() - start_time))
    plot_new(list(range(len(res))),res,"Results of penalty method",'exp_est.eps',r'$n$',r'$X_n$')
    #plt.show()

