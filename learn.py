import numpy as np
import tensorflow.keras as keras
import pickle

class Layers: #Custom Layer Class for each layer in the neural network
    def __init__(self ,prev_layer , neurons , activation ):
        self.neurons=  neurons #number of hidden features in the Layer
        self.prev_layer = prev_layer #number of hidden features in the previous Layer
        self.activation_name = activation #Activation function of the Layer
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
class Model:#custom model class
    def __init__(self , layers=[] , learning=0.05):
        self.layers = layers
        self.learning = learning
    def train(self, x_train , y_train , epochs , batch_size):
        
        for i in range(epochs):
            for iter in range(len(x_train) // batch_size):
                for b in range(batch_size):
                    delta = []
                    delta_bias = []
                    sample = np.random.randint(0 , len(x_train))
                    x = (x_train[sample].flatten()/255.).reshape(-1,1)
                    y = y_train[sample].reshape(-1,1)
                    
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
            print(f"Epoch: {i} has been completed")
            print(f"Loss : {1/2 * ((y - self.predict(x))**2)}")
    def predict(self,x):
        return self.layers[-1].predict(x)
    def add(self, layer):
        self.layers.append(layer)
    def save(self , name):

        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)
def preprocess(x):
    copy = np.tile(np.arange(0, 10), [len(x), 1])
    copy -= x.reshape(-1,1)
    return (copy== 0).astype(int)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x = np.random.normal( 0, 1, size=(784 , 1))
#print(x)
input_layer = Layers(None , 784    , 'relu')
second_layer = Layers( input_layer  , 256 , 'relu')
third_layer = Layers(second_layer , 128 , 'relu')
fourth_layer = Layers(third_layer , 10 , 'sigmoid')
model  =Model(learning=0.0001)
model.add(input_layer)
model.add(second_layer)
model.add(third_layer)
model.add(fourth_layer)
y_train = preprocess(y_train)
counter = 0
predictions = []

model.train(x_train , y_train , 10 , 32)
counter = 0
for i in x_test:
    result = model.predict(i.flatten().reshape(-1,1)/255)
    print(counter)
    predictions.append(int((np.argmax(result) == y_test[counter])))
    counter+=1
print("Accuracy : " , np.sum(predictions) / len(predictions))
model.save("mymodel.pkl")

#print(np.argmax(result))



