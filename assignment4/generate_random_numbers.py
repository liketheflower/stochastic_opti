# this is used to generate random number following uniform distribution

import numpy as np
def random_generation(iteration):
    a=np.random.rand(iteration)
    np.save('./random_number/random_'+str(iteration)+'.npy',a)
    b=np.random.rand(iteration)
    # generate another one for the MVD 
    print a[0],b[0]
    np.save('./random_number/y_random_'+str(iteration)+'.npy',b)

iterations = [2,100,1000,2000,5000,10000]
for iteration in iterations:
    random_generation(iteration)
