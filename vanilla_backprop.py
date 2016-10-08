import numpy as np

nn_input = 2
nn_output = 1
nn_hidden = 2
alpha = 0.8
#building Input Matrix
x = np.array(((1,1),(1,-1),(-1,1),(-1,-1)))

# building Output Matrix
t = np.array(((-1,), (1,), (1,) ,(-1,)))

def sigmoid(x):
    temp = np.tanh(x/2)
    temp[temp>=0] = 1
    temp[temp<0] = -1
    return temp

def tansig(x):
    return np.tanh(x/2)

def sigmoid_gradient(x):
    temp = 0.5 *(1+np.tanh(x))*(1-np.tanh(x))
    return temp

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    a2 = sigmoid(z2)
    q = len(t)
    cost = np.sum((z2 - t) ** 2) / q
    return cost

def build_model(iteration):
    # initializing weight and bais matrix
    w1 = np.random.uniform(low = -0.5,high = 0.5,size=(nn_input, nn_hidden))
    b1 = np.zeros((1, nn_hidden),dtype=float)
    w2 = np.random.uniform(low = -0.5,high = 0.5,size=(nn_hidden, nn_output))
    b2 = np.zeros((1, nn_output),dtype=float)

    model = {}
    for i in range(1,iteration):
        n1 = np.dot(x,w1) +b1
        a1= tansig(n1)
        n2 = np.dot(a1,w2)+b2
        a2 = sigmoid(n2)

        s2 = (n2-t)*0.5*(1-np.power(tansig(n2),2))
        dw2 = a1.T.dot(s2)
        s1 = s2.dot(w2.T)*0.5*(1-np.power(a1,2))
        dw1 = np.dot(x.T,s1)
        db2 = np.sum(s2,axis=0)
        db1 = np.sum(s1, axis=0)

        w2 = w2 - (alpha)*dw2
        b2 = b2-(alpha)*db2
        w1 = w1 - (alpha)*dw1
        b1 = b1 - (alpha)*db1

        model = {"w1": w1, "w2": w2, "b1": b1, "b2": b2}
        if i % 1 == 0:
            print "Loss after iteration %i: %f" % (i, calculate_loss(model))


    return model

def predict(model):
    w1,w2,b1,b2 = model['w1'],model['w2'],model['b1'],model['b2']
    n1 = np.dot(x, w1) + b1
    a1 = np.tanh(n1)
    n2 = np.dot(a1, w2) + b2
    a2 = sigmoid(n2)
    print a2

m = build_model(1000)
predict(m)
print m
