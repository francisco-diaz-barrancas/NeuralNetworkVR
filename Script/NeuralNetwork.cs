using UnityEngine;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Collections.Generic;
using System.Linq;
using System;

[System.Serializable]
public class NeuralNetwork
{

    int[] layer; // es el numero de capas y sus neuronas
    Layer[] layers; // las objetos de las capas
    private int sizeBatch ;
    private int numEpochs;
    private float learningRate;

    public NeuralNetwork(int[] layer, float learningRate,int sizeBatch,int numEpochs) // inicializa todo
    {
       
        this.layer = layer;
        this.learningRate = learningRate;
        this.sizeBatch = sizeBatch;
        this.numEpochs = numEpochs;
        layers = new Layer[layer.Length - 1];   // inicializa la neurnas pasandole el numero de neuronas y conexiones que debe de tener

        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(layer[i], layer[i + 1],learningRate);

        }
    }

    public float[] FeedForward(float[] inputs)
    {
        layers[0].FeedForward(inputs);
        for (int i = 1; i < layers.Length; i++)
        {
            layers[i].FeedForward(layers[i - 1].outputs);
        }
        return layers[layers.Length - 1].outputs;
    }

    public void BackProp(float[] expected)
    {
        for (int i = layers.Length - 1; i >= 0; i--)
        {
            if (i == layers.Length - 1)
            {
                layers[i].BackPropOutput(expected);
            }
            else
            {
                layers[i].BackPropHidden(layers[i + 1].errorS, layers[i + 1].weights);
            }
        }
    }

    public void UpdateNetwork()
    {
        for (int i = 0; i < layers.Length; i++)
        {
          
            layers[i].UpdateWeightsBias();
        }
    }
    
    public void SaveNetwork(string filename)
    {
        NeuralNetworkData networkData = new NeuralNetworkData
        {
            layer = this.layer,
            learning_rate = this.learningRate,
            sizeBatch = this.sizeBatch,
            num_epochs = this.numEpochs,
            layersData = new List<LayerData>()
        };

        foreach (Layer layer in this.layers)
        {
            LayerData layerData = new LayerData
            {
                biases = layer.getBiases(),
                weights = ConvertirMatrizAArray(layer.getWeights())
            };

            networkData.layersData.Add(layerData);
        }

        string jsonText = JsonUtility.ToJson(networkData, true);

        string filePath = Path.Combine(Application.persistentDataPath, "Results", filename);

        File.WriteAllText(filePath, jsonText);

        Debug.Log("Network saved successfully to: " + filePath);
    }
    public List<float> ConvertirMatrizAArray(float[,] matriz)
    {
        List<float> listaDeFloats = new List<float>();
        for (int i = 0; i < matriz.GetLength(0); i++)
        {
            for (int j = 0; j < matriz.GetLength(1); j++)
            {
                listaDeFloats.Add(matriz[i, j]);
                
            }
        }

        return listaDeFloats;
    }
    /*public void LoadNetwork(string filename)
    {
        string filePath = Path.Combine(Application.persistentDataPath, "Results", filename);

        if (!File.Exists(filePath))
        {
            Debug.LogError("El archivo de la red no existe en: " + filePath);
            return;
        }

        string jsonText = File.ReadAllText(filePath);

        NeuralNetworkData networkData = JsonUtility.FromJson<NeuralNetworkData>(jsonText);

        this.layer = networkData.layer;
        this.learningRate = networkData.learning_rate;
        this.sizeBatch = networkData.sizeBatch;
        this.numEpochs = networkData.num_epochs;

        this.layers = new Layer[layer.Length - 1];
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(layer[i], layer[i + 1], learningRate);

        }

        for (int i = 0; i < layers.Length; i++)
        {
            LayerData layerData = networkData.layersData[i];
            float[,] weights = ConvertirListaAMatriz(layerData.weights, layer[i], layer[i + 1]);
            

            // Establece los pesos y los sesgos de la capa
            this.layers[i].setLayerData(weights, layerData.biases);
            
        }
        
    }*/

    /*public float[,] ConvertirListaAMatriz(List<float> lista, int filas, int columnas)
    {
        if (lista.Count != filas * columnas)
        {
            throw new ArgumentException("La longitud de la lista no coincide con las dimensiones de la matriz.");
        }
        UnityEngine.Debug.Log("Filas:" + filas + ",Columnas: " + columnas);
        float[,] matriz = new float[filas, columnas];
        int index = 0;

        for (int i = 0; i < filas; i++)
        {
            for (int j = 0; j < columnas; j++)
            {
                matriz[i, j] = lista[index];
                //UnityEngine.Debug.Log(i + "x" + j);
                index++;
            }
        }
        
        return matriz;
    }*/
   

    public int getSizeBatch()
    {
        return this.sizeBatch;
    }
    public int getNumEpochs()
    {
        return this.numEpochs;
    }
    public float getLearningRate()
    {
        return this.learningRate;
    }
    public int [] getLayer()
    {
        return this.layer;
    }


    [System.Serializable]
    public class NeuralNetworkData
    {
        public int[] layer;
        public float learning_rate;
        public int sizeBatch;
        public int num_epochs;
        public List<LayerData> layersData;
    }

    [System.Serializable]
    public class LayerData
    {
        public float[] biases;
        public List<float> weights;
    }

    [System.Serializable]
    public class Layer
    {

        public int numberOfInputs; // numero de neuronas en la capa anterior
        public int numberOfOuputs;  // numero de neuronas en la capa actual
        public int numberOfPasses;

        public float[] outputs;
        public float[] inputs;
        public float[] biases;
        public float[] biasesDelta;
        public float[,] weights;
        public float[,] weightsDelta;
        public float[] errorS;
        public float[] error;
        public float learningRate;
        public Layer(int numberOfInputs, int numberOfOuputs,float learningRate)
        {
            this.numberOfInputs = numberOfInputs;
            this.numberOfOuputs = numberOfOuputs;

            outputs = new float[numberOfOuputs];
            inputs = new float[numberOfInputs];
            biases = new float[numberOfOuputs];
            biasesDelta = new float[numberOfOuputs];
            weights = new float[numberOfOuputs, numberOfInputs];
            weightsDelta = new float[numberOfOuputs, numberOfInputs];
            errorS = new float[numberOfOuputs];
            error = new float[numberOfOuputs];
            numberOfPasses = 0;
            this.learningRate = learningRate;
            InitilizeWeightsBias();
        }

        public void InitilizeWeightsBias()
        {
            for (int i = 0; i < numberOfOuputs; i++)
            {
                for (int j = 0; j < numberOfInputs; j++)
                {
                    weights[i, j] = UnityEngine.Random.Range(-1.0f, 1.0f);
                }
                biases[i] = UnityEngine.Random.Range(-0.5f, 0.5f);
            }
        }


        public float[] FeedForward(float[] inputs) // hace el todas las operaciones con un input que se le inserta
        {
            this.inputs = inputs;
            //UnityEngine.Debug.Log(numberOfInputs+","+numberOfOuputs);
            for (int i = 0; i < numberOfOuputs; i++)
            {
                outputs[i] = 0;
                for (int j = 0; j < numberOfInputs; j++)
                {
                    //UnityEngine.Debug.Log(i + "," + j);
                    outputs[i] += inputs[j] * weights[i, j]; //+ biases[i];
                }

                outputs[i] = sigmoid(outputs[i]);
                
            }

            return outputs;
        }
        

        public void BackPropOutput(float[] expected)
        {
            for (int i = 0; i < numberOfOuputs; i++)
                error[i] = outputs[i] - expected[i];

            for (int i = 0; i < numberOfOuputs; i++)
                errorS[i] = error[i] * sigmoidDelta(outputs[i]);

            for (int i = 0; i < numberOfOuputs; i++)
            {
                for (int j = 0; j < numberOfInputs; j++)
                {
                    weightsDelta[i, j] = errorS[i] * inputs[j];
                }
                biasesDelta[i] = errorS[i];
            }

            numberOfPasses++;
        }

        public void BackPropHidden(float[] errorS_forward, float[,] weights_forward)
        {
            for (int i = 0; i < numberOfOuputs; i++)
            {
                errorS[i] = 0;
                for (int j = 0; j < errorS_forward.Length; j++)
                {
                    errorS[i] += errorS_forward[j] * weights_forward[j, i];
                }

                errorS[i] *= sigmoidDelta(outputs[i]);
            }
            for (int i = 0; i < numberOfOuputs; i++)
            {
                for (int j = 0; j < numberOfInputs; j++)
                {
                    weightsDelta[i, j] = errorS[i] * inputs[j];
                }
                biasesDelta[i] = errorS[i];
            }

            numberOfPasses++;
        }

       
        public void UpdateWeightsBias()
        {
            for (int i = 0; i < numberOfOuputs; i++)
            {
                for (int j = 0; j < numberOfInputs; j++)
                {
                    weights[i, j] -= (weightsDelta[i, j] / numberOfPasses) /(1+ learningRate);
                }
                biases[i] -= (biasesDelta[i] / numberOfPasses) * 1.033f;
            }

            numberOfPasses = 0;
        }

        public float sigmoid(float x) // funcion sigmoid
        {
            return (1 / (1 + Mathf.Exp(-x)));
            //return (float)Math.Tanh(x);
        }

        public float sigmoidDelta(float x) // derivada de la funcion sigmoid
        {
            return sigmoid(x) * (1 - sigmoid(x));
            //return 1 - (x*x);
        }

        public float[,] getWeights()
        {
            return this.weights;
        }
        public float[] getBiases()
        {
            return this.biases;
        }

        public void setLayerData(float[,] weights, float[] biases)
        {
            this.weights = weights;
            this.biases = biases;
        }
    }
}
