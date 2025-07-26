import pickle
import cv2
import numpy as np
class Layers:
    def __init__(self ,prev_layer , neurons , activation ):
        self.neurons=  neurons
        self.prev_layer = prev_layer
        self.activation_name = activation
        if prev_layer is not None:
            self.weights  = np.random.randn(self.neurons, self.prev_layer.neurons) * np.sqrt(2. / self.prev_layer.neurons)
            self.bias = np.random.randn( self.neurons , 1) * np.sqrt(2. / self.prev_layer.neurons)
    def predict(self , x):
        if self.prev_layer is not None:
            return self.activation(self.weights @ self.prev_layer.predict(x) + self.bias  )
        else:
            return self.activation(x )
    def activation(self, x ):
        if self.activation_name == 'relu':
            return x*(x>0)
        else:
            return 1/(1+np.exp(-x))
    
    def derivative(self ,x ):
        if self.activation_name == 'relu':
            x = (x>0).copy()
            x[x == 0 ] = 0.01
            return x
        else:
            return  self.activation(x) * (1- self.activation(x))
class Model:
    def __init__(self , layers=[] , learning=0.05):
        self.layers = layers
        self.learning = learning
    def train(self, x , y , epochs):
        
        for i in range(epochs):
            delta = []
            delta_bias = []
            for j in self.layers[:0:-1]:
                
                gradient = y - self.predict(x)
                
                for k in self.layers[:self.layers.index(j):-1]:
                    gradient = ((gradient * k.derivative(k.weights @ self.layers[self.layers.index(k)-1].predict(x) + k.bias )).T @ k.weights).T
                bias = (gradient * j.derivative(j.weights @ self.layers[self.layers.index(j)-1].predict(x) + j.bias)) @ np.ones(shape=(1 , 1))
                gradient = (gradient * j.derivative(j.weights @ self.layers[self.layers.index(j)-1].predict(x) + j.bias)) @ self.layers[self.layers.index(j)-1].predict(x).T
                delta.append(gradient)
                delta_bias.append(bias)
            counter = 0
            for j in self.layers[:0:-1]:
                change = delta[counter]
                j.weights += self.learning * change
                j.bias += self.learning * delta_bias[counter]
                counter +=1    
            
    def predict(self,x):
        return self.layers[-1].predict(x)
    def add(self, layer):
        self.layers.append(layer)
    def save(self , name):

        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)
gray_image = cv2.imread('img_5.jpg', cv2.IMREAD_GRAYSCALE)
with open('mymodel3.pkl.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
result = loaded_model.predict((gray_image.flatten()/255.).reshape(-1,1))

print((np.argmax(result)))