'''
    The code is used to resolve the quene problem when the arrival of the customers follow the possion process.
    line search algo is used here.
    @author: jimmy shen
'''
import matplotlib.pyplot as plt
import numpy as np

def get_penalty(sita):
   return 1000000+2*sita
def get_cost(sita,b=10,alpha=0.01):
    if sita**b>alpha:
        return get_penalty(sita)
    else:
        return 1/(sita**2)

def line_search(iteration=1000):
    sita_cost=[]
    for i in range(iteration):
        sita=(i+1)/iteration
        sita_cost.append([sita,get_cost(sita)])
    return np.asarray(sita_cost)

def get_the_optimal_result(a):
    print ("The solution of (sita,cost) is:",a[a[:,1].argsort()][0])
def plot_2d(array_):
    #array_=np.asarray(list_)
    plt.scatter(array_[:,0], array_[:,1],alpha=0.8, c='#ca0020', s=3.5)
    plt.title('Sita v.s Cost when the Line Search algorithm is used')
    plt.ylabel("cost")
    plt.xlabel("sita)")
    plt.savefig("./output_method1.png",dpi = (200))
    plt.show()
if __name__=="__main__":
    sita_cost=line_search(1000)
    get_the_optimal_result(sita_cost)
    plot_2d(sita_cost)
