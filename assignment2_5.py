'''
    The code is used to resolve the exercise 2.5 of stochastic optimization course.
    line search algo is used here.
    @author: jimmy shen
'''
import matplotlib.pyplot as plt
import numpy as np
def plot_2d(x,y,x_lab,y_lab,title_,file_name,plot_line=False):
    #array_=np.asarray(list_)
    plt.scatter(x, y, s=3.5)
    if plot_line:
        plt.plot(x,y,'-')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title_)
    
    plt.savefig(file_name,dpi = (200))
    plt.close()
#plt.show()

def get_deri(sita,a,b,d,v1,v2):
    return (1/v1)*sita/((sita**2+a**2)**0.5)-(1/v2)*(d-sita)/(((d-sita)**2+b**2)**0.5)
if __name__=="__main__":
    a = 2
    b = 5
    d = 10
    v1 = 3
    v2 = 1
    step=0.05
    results=[]
    sita = 0.5
    for i in range(1000):
        dj = get_deri(sita,a,b,d,v1,v2)
        sita-=step*dj
        results.append([i,sita])
    print (results[-1])
    results= np.asarray(results)


    x_lab = "Iteration"
    y_lab = "Sita"
    title_ = "Iteration v.s Sita for Exercise 2.5"
    file_name = "./assignment1_2_5.png"

    plot_2d(results[:,0],results[:,1],x_lab,y_lab,title_,file_name)
