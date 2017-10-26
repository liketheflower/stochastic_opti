'''
    The code is used to resolve the 4.3b of the chapter 4 of the stocahstic optimization provided at the Graduate Center.
    Oct 25, 2017
    @author: jimmy shen
'''
import matplotlib.pyplot as plt
import numpy as np
def plot_2d(x,y,N,plot_id):
    #array_=np.asarray(list_)
    plt.scatter(x,y,alpha = 0.03)
    plt.title(r'$\theta_n$'+" V.S. " +r'$n$'+' when N = '+str(N)+", with "+r'$\epsilon_n = 1/n $')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\theta _n$')
    plt.ylim( 0.3, 1.0 )
    #plt.savefig("./output"+str(plot_id)+".png",dpi = (200))
    plt.show()
    plt.close()
#plt.show()

def plot_2d_heart_rate(x,y,N,plot_id):
    #array_=np.asarray(list_)
    plt.scatter(x,y,alpha = 0.03)
    plt.title('Expected Heart Rate'+" V.S. " +r'$n$'+' when N = '+str(N)+", with "+r'$\epsilon_n = 1/n $')
    plt.xlabel(r'$n$')
    plt.ylabel('Expected Heart Rate')
    plt.ylim( 60, 200 )
    plt.savefig("./expected_heart_rate"+str(plot_id)+".png",dpi = (200))
    
    plt.close()

def get_the_random_numbers_used_in_the_simulation(numbers):
    np.random.seed(0)
    return np.random.rand(numbers)

def do_simulation(theta_n,heart_beat_n,learning_rate,randoms,N,target_heart_beat,plot_id):
    # when n=0, the result is already initialized, so the simulation begins from the second measure interval
    for i in range(1,heart_beat_n.shape[0]):
        #generate_simulated_heart_beat
        if heart_beat_n[i-1] ==0:
            #the probability of changing the heart beat from 0 to 0 is writen as p00
            p01 = theta_n[i-1]
            #p00 = 1-p01
            if randoms[i]<=p01:
                heart_beat_n[i] = 1
            else:
                heart_beat_n[i] = 0
        else:
            p11 = theta_n[i-1]**2
            #p10 = 1- p11
            if randoms[i]<=p11:
                heart_beat_n[i] = 1
            else:
                heart_beat_n[i] = 0
        
        #update_theta
        if i%N ==0:
            sum_of_heart_beat_in_N_intervals = heart_beat_n[i-N:i].sum()
            theta_n[i] = theta_n[i-1]-learning_rate[i]*(200*sum_of_heart_beat_in_N_intervals-N*target_heart_beat)
            protection = True
            if protection:
                if theta_n[i]<0.5:
                    theta_n[i] = 0.5
                if theta_n[i]>1.0:
                    theta_n[i] = 1.0
        #theta_n[i] = theta_n[i-1]+learning_rate[i]*(200*sum_of_heart_beat_in_N_intervals-N*target_heart_beat)
        else:
            theta_n[i]=theta_n[i-1]
    plot_2d(np.arange(heart_beat_n.shape[0]),theta_n,N,plot_id)
    expected_heart_rate = np.zeros(theta_n.shape)
    for m in range(theta_n.shape[0]):
          expected_heart_rate[m] = 200*(theta_n[m]/(1+theta_n[m]*(1-theta_n[m])))
    plot_2d_heart_rate(np.arange(heart_beat_n.shape[0]),expected_heart_rate,N,plot_id)
    return theta_n,heart_beat_n

if __name__=="__main__":
    #total number of the measure intervals. Each interval is 0.3s then the total simulation time will be 0.3*TOTAL_NUMBER
    TOTAL_NUMBER = 10000
    randoms = get_the_random_numbers_used_in_the_simulation(TOTAL_NUMBER)
    print (randoms.shape)
    theta_n = np.zeros((TOTAL_NUMBER,),dtype=float)
    heart_beat_n = np.zeros((TOTAL_NUMBER,),dtype=int)
    #initialize theta0 as 0.5, initialize the first measure of the heart beat as 0.
    theta_n[0] = 0.5
    heart_beat_n[0] = 0
    learning_rate = np.ones((TOTAL_NUMBER,))
    for m in range(learning_rate.shape[0]):
        learning_rate[m] =float(1.0/((m+1))**(1/1.0))
    N = [1,10,20]
    target_heart_beat = 120
    plot_id = 100
    for k in range(0,3):
        theta_n,heart_beat_n = do_simulation(theta_n,heart_beat_n,learning_rate,randoms,N[k],target_heart_beat,plot_id)
        plot_id+=1

#plot_2d(np.arange(TOTAL_NUMBER),heart_beat_n,100000,plot_id+9)
