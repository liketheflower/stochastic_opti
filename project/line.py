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

def get_j_a_n(x_n,a):
    if x_n <=1 and x_n>=0:
        return -1
    elif x_n<0:
        return -1+a*x_n
    else:
        return -1+a*x_n


def simulation(iteration, learning_rate, alpha_n):
    res=[]
    smooth =10
   # estimate_x=[]
 
    x_init=20 
    for i, a in enumerate(alpha_n):
        print 'i,a',i,a
        tem = []
        for i in range(iteration):
            if i==0:
                tem.append(x_init)
            tem.append(tem[i-1]-learning_rate*get_j_a_n(tem[i-1],a))
        res+=tem
        average_ = sum(tem[-smooth:])/float(smooth)
        if 1000*average_>=0 and 1000*average_<=1000: 
            return res , average_
    return res, average_


if __name__=='__main__':
    iteration = 500
    learning_rate = 0.01
    penalty_max_n = 100 
    alpha_n = [math.exp(i) for i in range(penalty_max_n)]
    res,estimated_x = simulation(iteration, learning_rate, alpha_n)
    print 'res', res[-1],estimated_x
    plt.plot(list(range(len(res))),res)
    plt.show()

