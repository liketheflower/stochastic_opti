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
    def plot_line(self):
        # a simple line:  y = 2*x +1*(1-x) =2x+1-x =x+1
        y = [x_+1 for x_ in self.x]
        self.plot_(y,r'$J(x)=-x-1$','line.eps')
    def plot_line_sphe(self):
        #   y = 2*cos(x)**2 + (1-cos(x)**2)
        #   y = cos(x)**2 + 1
        y = [math.cos(x_)**2+1 for x_ in self.x]
        self.plot_(y,r'$J(\theta) = -cos(\theta)^2-1$','line_sphe.eps',r'$\theta$',r'$J(\theta)$')
    
    def plot_cos_x_square(self):
        y = [math.cos(x_)**2 for x_ in self.x]
        self.plot_(y) 
    def plot_poly_order_4(self):
        y = [x_**4-2*x_**2+x_ for x_ in self.x]
        self.plot_(y)

    def plot_poly_order_4_sphe(self):
        y = [math.cos(x_)**8-2*math.cos(x_)**4+math.cos(x_) for x_ in self.x]
        self.plot_(y)
    def plot_quartic_curve(self):
        # y = x**2 + (1-x)**2
        y = [x_**2+(1-x_)**2 for x_ in self.x]
        self.plot_(y)

    def plot_quartic_curve_sphe(self):
        # y = x**2 + (1-x)**2
        y = [math.cos(x_)**4+(1-math.cos(x_)**2)**2 for x_ in self.x]
        self.plot_(y)
    def exponential(self):
        c = 0.01
        y = [c*math.exp(x_) for x_ in self.x]
        self.plot_(y)
    def get_log_norm_pdf(self,x,mu=1.0,sigma=0.0):
        return math.exp(    -(math.log(x)-mu)**2/(2*sigma**2)       )/(x*sigma*(2*math.pi)**0.5)
    def log_norm(self):
        # ref:  http://mathworld.wolfram.com/LogNormalDistribution.html
        mu,sigma = 0, 1	
        y = [self.get_log_norm_pdf(x_, mu,sigma) for x_ in self.x]
        self.plot_(y)
    def log_norm_and_exponential(self):
        
        mu,sigma = 0, 1	
        c = 0.01
        y = [self.get_log_norm_pdf(x_, mu,sigma)+c*math.exp(x_) for x_ in self.x]
        self.plot_(y)
a = Simple(0.01,-math.pi,4*math.pi)
a.plot_line()
a.plot_line_sphe()
"""
#a.plot_quartic_curve()
b = Simple(0.001, -6, 6)
b.plot_quartic_curve()
b.plot_quartic_curve_sphe()

c = Simple(0.01, -1.5, 1.5)
c.plot_poly_order_4()
"""




"""
d = Simple(0.01, 0.01, 4*math.pi)
d.plot_poly_order_4_sphe()
d.exponential()
d.log_norm()
d.log_norm_and_exponential()

"""
