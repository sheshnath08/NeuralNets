import  numpy as np


def computeCost(y, t):
    q= len(t)
    cost = np.sum((y-t)**2)/q
    return cost

#building Input Matrix
s = np.array(((1,1,1),(1,2,0),(2,-1,-1),(2,0,0),(-1,2,-1),(-2,1,1),(-1,-1,-1),(-2,-2,0)))

# building Output Matrix
t = np.array(((-1,-1),(-1,-1),(-1,1),(-1,1),(1,-1),(1,-1),(1,1),(1,1)))

#building input vector by appending bais input and original input
x = np.append(np.ones([len(s),1]),s,1)

#learning Rate alpha
alpha = 0.04

#initial Weights and Bais
w = np.array(((0,0),(0,0),(0,0),(0,0)))

yin = np.dot(x,w)
change_in_weight = 100
for i in range(0,40):
    yin = np.dot(x,w)
    y = yin
    w = w-(2*alpha)*np.dot(x.T,(y-t))
    #print w
    print computeCost(y,t)

#print np.dot(x,w)