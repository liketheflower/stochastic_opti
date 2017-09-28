'''
    The code is used to resolve the quene problem when the arrival of the customers follow the possion process.
    line search algo is used here.
    @author: jimmy shen
'''
import matplotlib.pyplot as plt
import numpy as np

def get_penalty(sita):
    return 1000000+1000000*sita
def get_cost(sita,b=10,alpha=0.01):
    if sita**b>alpha:
        return get_penalty(sita)
    else:
        return 1/(sita**2)

def get_gradient(sita,b=10,alpha=0.01):
    if sita**b>alpha:
        #the penalty function is 1000000+1000000*sita
        return 2
    else:
        return -2*(1/sita**3)

def gradient_descent(sita=0.5,iteration=1000):
    iteration_sita_cost= []
    #cost= []
    
    for i in range(iteration):
        # step size is reduced when the iteration increases
        step_size = 0.01/(1+i)
        gradient = get_gradient(sita)
        sita -=step_size*gradient
        #sita=(i+1)/iteration
        iteration_sita_cost.append([i,sita,get_cost(sita)])
    
    return np.asarray(iteration_sita_cost)

def get_the_optimal_result(a):
    print ("The solution of (iteration, sita,cost) is:",a[a[:,2].argsort()][0])
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
if __name__=="__main__":
    iteration_sita_cost=gradient_descent(0.5,120)
    get_the_optimal_result(iteration_sita_cost)
    x_lab = "iteration"
    y_lab = "sita"
    title_ = "Iteration v.s Sita when the Gradient Descend algorithm is used"
    file_name = "./output_method2_1.png"
    plot_2d(iteration_sita_cost[:,0],iteration_sita_cost[:,1],x_lab,y_lab,title_,file_name)
    x_lab = "iteration"
    y_lab = "cost"
    title_ = "Iteration v.s Cost when the Gradient Descend algorithm is used"
    file_name = "./output_method2_2.png"

    plot_2d(iteration_sita_cost[:,0],iteration_sita_cost[:,2],x_lab,y_lab,title_,file_name)
    x_lab = "Sita"
    y_lab = "Cost"
    title_ = "Sita v.s Cost when the Gradient Descend algorithm is used"
    file_name = "./output_method2_3.png"

    plot_2d(iteration_sita_cost[:,1],iteration_sita_cost[:,2],x_lab,y_lab,title_,file_name, True)

