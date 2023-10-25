import numpy as np
import time
import pickle
class NeuralNetwork:
    def __init__(self, layer, learning_rate):
        self.layer = layer
        self.learning_rate = learning_rate
        self.layers = [Layer(layer[i], layer[i + 1], learning_rate) for i in range(len(layer) - 1)]

    def feed_forward(self, inputs):
        self.layers[0].feed_forward(inputs)
        for i in range(1, len(self.layers)):
            self.layers[i].feed_forward(self.layers[i - 1].outputs)
        return self.layers[len(self.layers)-1].outputs

    def back_prop(self, expected):
        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                self.layers[i].back_prop_output(expected)
            else:
                self.layers[i].back_prop_hidden(self.layers[i + 1].errorS, self.layers[i + 1].weights)

    def update_network(self):
        for i in range(len(self.layers)):
            self.layers[i].update_weights_bias()
            
    def save_network(self, filename):
        network_data = {
            'layer': self.layer,
            'learning_rate': self.learning_rate,
            'layers_data': [layer.get_layer_data() for layer in self.layers],
        }
        with open(filename, 'wb') as file:
            pickle.dump(network_data, file)

    @classmethod
    def load_network(cls, filename):
        with open(filename, 'rb') as file:
            network_data = pickle.load(file)

        network = cls(network_data['layer'], network_data['learning_rate'])
        for i in range(len(network.layers)):
            network.layers[i].set_layer_data(network_data['layers_data'][i])

        return network        

class Layer:
    def __init__(self, numberOfInputs, numberOfOutputs, learningRate):
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.outputs = np.zeros(numberOfOutputs)
        self.inputs = np.zeros(numberOfInputs)
        self.biases = np.random.uniform(-0.5, 0.5, numberOfOutputs)
        
        self.biasesDelta = np.zeros(numberOfOutputs)
        self.weights = np.random.uniform(-1.0, 1.0, (numberOfOutputs, numberOfInputs))
        
        self.weightsDelta = np.zeros((numberOfOutputs, numberOfInputs))
        self.errorS = np.zeros(numberOfOutputs)
        self.error = np.zeros(numberOfOutputs)
        self.learningRate = learningRate
        self.numberOfPasses = 0
        self.delta = None  # Nuevo atributo para almacenar los valores de delta
        
    def feed_forward(self, inputs):
        self.inputs = inputs
        outputs = np.dot(self.weights, inputs)  # Reemplace con operaciones vectorizadas
        outputs = self.sigmoid(outputs)
        self.outputs = outputs
        return outputs
        
    def back_prop_output(self, expected):
        self.error = self.outputs - expected
        self.errorS = self.error * self.sigmoid_delta(self.outputs)
        
        self.weightsDelta = np.outer(self.errorS, self.inputs)
        self.biasesDelta = self.errorS
        
        self.numberOfPasses += 1

    def back_prop_hidden(self, errorS_forward, weights_forward):
        self.errorS = np.dot(errorS_forward, weights_forward) * self.sigmoid_delta(self.outputs)
    
        self.weightsDelta = np.outer(self.errorS, self.inputs)
        self.biasesDelta = self.errorS
        
        self.numberOfPasses += 1

    def update_weights_bias(self):
        self.weights -= (self.weightsDelta / self.numberOfPasses) / (1+self.learningRate)
        self.biases -= (self.biasesDelta / self.numberOfPasses) * 1.033

        self.numberOfPasses = 0

    def sigmoid(self, x):
        x = x.astype(float)  # Asegurarse de que los valores de entrada sean de tipo float
     
        return 1 / (1 + np.exp(-x))

    def sigmoid_delta(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def get_layer_data(self):
        return {
            'weights': self.weights,
            'biases': self.biases
        }

    def set_layer_data(self, layer_data):
        self.weights = layer_data['weights']
        self.biases = layer_data['biases']