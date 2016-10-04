import  numpy as np

def computeCost(y, t):
    q= len(t)
    cost = np.sum((y-t)**2)/q
    return cost

#building Input Matrix
s = np.array(((1,1,1),(1,2,0),(2,-1,-1),(2,0,0),(-1,2,-1),(-2,1,1),(-1,-1,-1),(-2,-2,0)))

# building Output Matrix
t = np.array(((-1,-1), (-1,-1), (-1,1) ,(-1,1), (1,-1), (1,-1), (1,1), (1,1)))

#building input vector by appending bais input and original input
x = np.append(np.ones([len(s),1]),s,1)

#learning Rate alpha
alpha = 1.0

#initial Weights and Bais
w = np.array(((0,0),(0,0),(0,0),(0,0)))


for i in range(1,50):
    yin = np.dot(x,w)
    y = yin
    alpha = 1.0/i
    w = w-(2*alpha)*np.dot(x.T,(y-t))

print "Final Error is: ",computeCost(y,t)
print "Final Alpha is: ",alpha
print "Final Bais is: ",w[0:1,:]
print "Final Weight is: "
print w[1:,:]
op = np.dot(x,w)
final_output = np.ones_like(op,dtype=int)
final_output[op<0] = -1
print "Input is classified as: "
print final_output
