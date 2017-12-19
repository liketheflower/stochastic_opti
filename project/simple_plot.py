# this code is used to generate a simle plot of : y=x**2+(1-x)**2
# by jimmy shen 
# Dec 12, 2017

import numpy as np
import matplotlib.pyplot as plt
class Simple(object):
    def __init__(self):
        print "this is a simple example"
    def plot_(self, step, lower, upper):
        if step>1.0 or step <0.0:
            print "error: step has to be greater than 0 and less than 1.0"
        x = [i for i in range(int(lower/step),int(upper/step))]
        y = [x_**2+(1-x_)**2 for x_ in x]
        plt.plot(x,y)
        plt.show()

a = Simple()
a.plot_(0.1,-5,5)

