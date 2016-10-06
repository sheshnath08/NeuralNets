import numpy as np

def sigmoid(x):
    temp = np.tanh(x/2)
    print temp
    if temp>=0:
        return 1
    else:
        return -1

def computeCost(y, t):
    cost = np.sum((y - t) ** 2) / 2.0
    return cost

#building Input Matrix
s = np.array(((1,1),(1,-1),(-1,1),(-1,-1)))

# building Output Matrix
t = np.array(((-1), (1), (1) ,(-1)))

#building input vector by appending bais input and original input
x = np.append(np.ones([len(s),1]),s,1)

#initializing weight and bais matrix
w1 = np.random.uniform(low= -0.5,high=0.5,size=(3,2))
w2 = np.random.uniform(low=-0.5,high= 0.5, size=(2,1))
for i in range(0,10):
    y1 = np.dot(x,w1)
    y2 = np.dot(y1,w2)

print y2

