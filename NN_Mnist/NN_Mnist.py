import numpy as np
np.random.seed(0)

X = [[2,9], [3,7], [5,8], [1,4], [4,5], [1,1], [12,11], [11,9], [9,8], [10,6], [13,6], [11,2]]
X = np.array(X)
#y = [[0], [1], [0], [0], [1], [0], [1], [1], [0], [0], [1], [1]]
y = [[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]]
y = np.array(y)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.itea = 0.1
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    def sig(self):
        self.output = np.multiply(self.output, -1)
        self.output = np.exp(self.output)
        self.output = np.add(1, self.output)
        self.output = np.divide(1, self.output)
    def error(self):
        self.error = np.subtract(y,self.output)
        self.error = np.multiply(self.error, self.error)
        self.error = np.multiply(self.error,0.5)
    def backward_output(self, layer):
        # ^wji = itea *((tj - oj)oj(1-oj) , oi)
        w_new = np.subtract(y, self.output)
        one_oj = np.subtract(1,self.output)
        out_j = np.multiply(self.output,one_oj)
        self.eq = np.multiply(w_new,out_j) # Sigma K
        ww = np.dot(layer.output.T,self.eq)#12*1  12*4 1 * 4   12*1  4 * 12
        ww_new = np.multiply(self.itea,ww)
        self.weights += ww_new
    def hidden_back(self,layer,X):
        half_eq = np.dot(layer.eq,layer.weights.T) #12 * 1 , 4 * 1
        one_oj = np.subtract(1, self.output)
        out_j = np.multiply(self.output, one_oj)
        segma_j = np.multiply(half_eq,out_j)
        ww = np.dot(X.T,segma_j) #  12 * 2 , 12 * 4
        ww_new = np.multiply(self.itea, ww)
        self.weights += ww_new
    def predict(self,X,layer):
        self.forward(X)
        self.sig()
        layer.forward(self.output)
        layer.sig()
        print(f"Neural Network Prediction Ratio: {layer.output * 100}")
        if(np.abs(1 - layer.output) < np.abs(0 - layer.output)):
            print("Predicted class : 1")
        else:
            print("Predicted class : 0")

layer1 = Layer_Dense(2,4)
layer2 = Layer_Dense(4,1)

for i in range(100):
    layer1.forward(X)
    layer1.sig()
    layer2.forward(layer1.output)
    layer2.sig()
    layer2.backward_output(layer1)
    layer1.hidden_back(layer2,X)
 #14,4 #2,4

layer1.predict([14,4],layer2)
layer1.predict([2,4],layer2)
