"""
Group:

    Jan-Erik Huehne
    Oscar Soto
    Joseph Gonzalez
"""
import numpy as np
import pickle

class NN(object):
    def __init__(self):
        self.layers = []
        self.input = None
        self.output = None
        self.is_build = False
        self.pitch_min_norm = None
        self.roll_min_norm = None
        self.pitch_max_norm = None
        self.roll_max_norm = None

    def __call__(self, l):
        self.layers.append(l)
    
    def __build__(self, input_dim):
        """
        This method is used to build the NN, by initalizing its weight matrices in the different layers.
        """
        self.input= self.layers[0]
        self.output = self.layers[-1]
        self.layers[0].__build__(input_dim)
        for i, l in enumerate(self.layers[1:]):
            l.__build__(self.layers[i].units)
        self.is_build = True

    def predict(self, x):
        result = x
        for l in self.layers:
            result = l.forward(result)
        
        return result 

    def train(self, X_train, Y_train, X_test, Y_test,  Optimzier, loss_func, batch_size=16, epochs=20):
        # First we build the neural network
        if not self.is_build:
            print("Building with input dim ", X_train.shape[0])
            self.__build__(X_train.shape[0])
        # We get the number of batches per epoch 
        batch_number  = -(-X_train.shape[-1] // batch_size)
        # We can estimate the inital loss of the network
        train_prediction = self.predict(X_train)
        train_cost = loss_func.forward(Y_train, train_prediction)
        print("Inital Loss: {} ".format(round(train_cost,6)))
        for i in range(epochs):
            # Shuffle the data
            permutation = np.random.permutation(X_train.shape[1])
            X_train_shuffled = X_train[:, permutation]
            Y_train_shuffled = Y_train[:, permutation]
            for j in range(batch_number):
                # Get the next batch of data
                begin = j * batch_size
                end = min(begin + batch_size, X_train.shape[1] - 1)
                X_b = X_train_shuffled[:, begin:end]
                Y_b = Y_train_shuffled[:, begin:end]
                # Forward Pass
                y_hat = self.predict(X_b)  
                # Get Loss Function Backward Delta
                delta_0 = loss_func.backward(Y=Y_b, Y_hat=y_hat)  
                # Backpropagate errors
                self.backprop(error=delta_0)
                # make a step with the optimizer
                Optimzier.step(i+1)
                
            # Calculate the loss
            train_prediction = self.predict(X_train)
            train_cost = loss_func.forward(Y_train, train_prediction)
            if X_test != None and Y_test != None: 
                test_prediction = self.predict(X_test)
                test_cost = loss_func.forward(Y_test, test_prediction)
            else: 
                test_cost = 0.0
            print("Epoch {}: training cost: {}".format(i+1 ,round(train_cost,6)))
        

    def backprop(self, error):
        er = error
        for l in reversed(self.layers):
            er = l.backward(er)


class CrossEntropyLoss():

    def forward(self, Y, Y_hat):
        Y_hat = self._softmax(Y_hat)
        L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
        m = Y.shape[1]
        L = -(1./m) * L_sum
        return L
    
    def  _softmax(self, x):
        return np.exp(x- np.max(x, axis=0)) / np.sum(np.exp(x - np.max(x, axis=0)), axis=0)
    
    def backward(self,y, y_hat):
        y_hat = self._softmax(y_hat)
        return y_hat - y

class MSELoss():
    def forward(self, Y, Y_hat):
        return np.mean((Y-Y_hat)**2)
    
    def backward(self, Y, Y_hat):
        return -2.0 * (Y-Y_hat)
    
class RMSProp(object):
    """
    Version of RMSProp Optimizer
    """
    def __init__(self, model, beta=0.9, lr=0.01):
        self.model = model
        self.beta = beta
        self.lr = lr 

    def step(self):
        for layer in self.model.layers:
            if layer.trainable:
                layer.VW = self.beta * layer.VW 
                layer.Vb = self.beta * layer.Vb
                layer.VW += (1. - self.beta) * layer.dW
                layer.Vb += (1. - self.beta) * layer.db
                layer.W = layer.W - self.lr * layer.VW 
                layer.b = layer.b - self.lr * layer.Vb 

class Adam(object):
    def __init__(self, model, beta=0.9, lr=0.01):
        self.model = model
        self.beta1 = 0.9
        self.beta2 = 0.9
        self.epsilon = 1e-8
        self.eta = lr

    def step(self, t):

        for layer in self.model.layers:
            """
            If the layer is trainable, we apply the ADAM update rule
            """
            if layer.trainable:
                layer.m_dw = self.beta1*layer.m_dw + (1-self.beta1)*layer.dW
                layer.v_dw = self.beta2*layer.v_dw + (1-self.beta2)*(layer.dW**2)
                m_dw_corr = layer.m_dw/(1-self.beta1**t)
                v_dw_corr = layer.v_dw/(1-self.beta2**t)
                layer.W  -= self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))

                layer.m_db = self.beta1*layer.m_db + (1-self.beta1)*layer.db
                layer.v_db = self.beta2*layer.v_db + (1-self.beta2)*(layer.db**2)
                m_db_corr = layer.m_db/(1-self.beta1**t)
                v_db_corr = layer.v_db/(1-self.beta2**t)
                layer.b  -= self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
            
class Linear(object):
    def __init__(self, units):
        self.trainable = True
        self.units = units
        self.W = None
        self.dW = None
        self.db = None

        self.m_dw = None
        self.m_db = None
        self.v_dw = None
        self.v_db = None

        self.cache = {}

    def __build__(self, input_dim):
        # weight matrix has the shape of  N_input x  N_output
        
        self.W = np.random.randn(self.units,input_dim) / np.sqrt(input_dim)
        self.b = np.zeros((self.units, 1))  
        self.VW = np.zeros(self.W.shape)
        self.Vb = np.zeros(self.b.shape)

        self.m_dw = np.zeros(self.W.shape)
        self.m_db = np.zeros(self.b.shape)
        self.v_dw = np.zeros(self.W.shape)
        self.v_db = np.zeros(self.b.shape)

    def forward(self, X):
        # X has the shape of N_batch * N_input 
        self.cache["input"] = X
        return  np.matmul(self.W, X) + self.b
       

    def backward(self, delta):
        # delta is of shape N_batch x N_output

        # We need to get the shape of N_input x N_output for the weight matrix 
        # So we take X^T * delta 
        factor = 1. / delta.shape[0]
        self.dW = factor * (np.matmul(delta,self.cache["input"].T ))
        self.db = factor * np.sum(delta, axis=1, keepdims=True)
       
        return np.matmul(self.W.T, delta)



class Sigmoid(object):
    def __init__(self, units):
        self.trainable = False
        self.units = units
        self.cache = {}
    
    def _sigmoid(self, x):
        return 1  / (1 + np.exp(-x))
    
    def __build__(self, *args,  **kwargs):
        pass

    def forward(self, x):
        self.cache["input"] = x
        return self._sigmoid(x)

    def backward(self, delta):
        return delta * self._sigmoid( self.cache["input"]) * (1 - self._sigmoid(self.cache["input"]))
    
class ReLU(object):
    def __init__(self, units):
        self.trainable = False
        self.units = units
        self.cache = {}
    def forward(self, X):
        self.cache["input"] = X
        X[X<0] = 0 
        return X
    def __build__(self, *args,  **kwargs):
        pass
    def backward(self, delta):
        mask = self.cache["input"]
        mask[mask < 0] = 0 
        mask[mask > 0] = 1
        return delta * mask 

if __name__ == "__main__":
    
    # We define our neural network
    nn = NN()
    nn(Linear(40))
    nn(ReLU(40))
    nn(Linear(30))
    nn(ReLU(30))
    nn(Linear(2))
    nn(Sigmoid(2))

    # Load the training data 
    with open("data2.txt") as f:
        samples = f.readlines()
    # Extract the training samples(red blob positions) and labels (shoulder joint positions)
    data_samples = []
    labels = []
    for sample in samples:
        s = sample.split(" ")
        data_samples.append([float(s[0]), float(s[1])])
        labels.append([float(s[2]), float(s[3])])
    data_samples = np.array(data_samples)
    labels = np.array(labels)

    # We normalize the training samples 
    data_samples = (data_samples - np.min(data_samples)) / (np.max(data_samples)- np.min(data_samples))

   
    # We get the values needed for normalization (Normalizing each input axis seperatly)
    pitch_min_norm = np.min(labels, axis=0)[0]
    roll_min_norm = np.min(labels, axis=0)[1]
    pitch_max_norm = np.max(labels, axis=0)[0]
    roll_max_norm = np.max(labels, axis=0)[1]
    nn.pitch_min_norm = pitch_min_norm
    nn.pitch_max_norm = pitch_max_norm
    nn.roll_min_norm = roll_min_norm
    nn.roll_max_norm = roll_max_norm
    
    nlabels = labels.copy()
    nlabels[:, 0] = (labels[:, 0]-pitch_min_norm) / (pitch_max_norm - pitch_min_norm)
    nlabels[:, 1] = (labels[:, 1]-roll_min_norm) / (roll_max_norm - roll_min_norm)
  
    opt = Adam(model=nn,lr=0.05)
    lf = MSELoss()
    # Train the neural network
    nn.train(data_samples.T, nlabels.T, None, None, opt, lf, batch_size=10, epochs=200)
    
    # Save the trained neural network instance as pickle file
    with open("nn.pkl", "wb") as f:
        pickle.dump(nn, f, pickle.HIGHEST_PROTOCOL)
    
    # Get additonal information on the performance of the trained network 
    pred = nn.predict(data_samples.T).T

    pred[:, 0] = (pred[:, 0]* (pitch_max_norm - pitch_min_norm) ) + pitch_min_norm 
    pred[:, 1] = (pred[:, 1]* (roll_max_norm - roll_min_norm)) + roll_min_norm

    for i in range(len(pred)):
        print("Predicted: {:.2f}  {:.2f}   True: {:.2f}  {:.2f}   Difference: {:.2f}  {:.2f}".format(pred[i,0], pred[i,1], labels[i,0], labels[i,1],pred[i,0]-labels[i,0],pred[i,1]-labels[i,1]))
        