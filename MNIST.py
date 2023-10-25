import numpy as np
from NeuralNetwork import NeuralNetwork
import random
import time
import psutil
import os
# Parámetros de configuración
sizeBatch = 8
num_epochs = 10
neurona_start = 120
neurona_intermedia = 120
learning_rate = 0.1
valores_de_entrenamiento = None
etiquetas_entrenamiento = None
valores_de_prueba = None
etiquetas_prueba = None

# Función para inicializar datos de entrenamiento
def inicializar(label_nombre, images_nombre, num_datos):
    valores_de_entrenamiento = np.zeros((num_datos, 784), dtype=float)
    etiquetas_entrenamiento = np.zeros(num_datos, dtype=int)
    
    with open(label_nombre, 'rb') as labels_file, open(images_nombre, 'rb') as images_file:
        magic1 = int.from_bytes(images_file.read(4), byteorder='big')
        num_images = int.from_bytes(images_file.read(4), byteorder='big')
        num_rows = int.from_bytes(images_file.read(4), byteorder='big')
        num_cols = int.from_bytes(images_file.read(4), byteorder='big')

        magic2 = int.from_bytes(labels_file.read(4), byteorder='big')
        num_labels = int.from_bytes(labels_file.read(4), byteorder='big')

        pixels = np.zeros((28, 28), dtype=float)

        for di in range(60000):
            for i in range(28):
                for j in range(28):
                    b = int.from_bytes(images_file.read(1), byteorder='big')
                    pixels[i][j] = b/255
                    
                    valores_de_entrenamiento[di][(28 * i) + j] = pixels[i][j]

            lbl = int.from_bytes(labels_file.read(1), byteorder='big')
            etiquetas_entrenamiento[di] = lbl

    return valores_de_entrenamiento, etiquetas_entrenamiento

# Función para inicializar datos de entrenamiento
def inicializarPrueba(label_nombre, images_nombre, num_datos):
    valores_de_prueba = np.zeros((num_datos, 784), dtype=float)
    etiquetas_prueba = np.zeros(num_datos, dtype=int)
    
    with open(label_nombre, 'rb') as labels_file, open(images_nombre, 'rb') as images_file:
        magic1 = int.from_bytes(images_file.read(4), byteorder='big')
        num_images = int.from_bytes(images_file.read(4), byteorder='big')
        num_rows = int.from_bytes(images_file.read(4), byteorder='big')
        num_cols = int.from_bytes(images_file.read(4), byteorder='big')

        magic2 = int.from_bytes(labels_file.read(4), byteorder='big')
        num_labels = int.from_bytes(labels_file.read(4), byteorder='big')

        pixels = np.zeros((28, 28), dtype=float)

        for di in range(10000):
            for i in range(28):
                for j in range(28):
                    b = int.from_bytes(images_file.read(1), byteorder='big')
                    pixels[i][j] = b/255
                    
                    valores_de_prueba[di][(28 * i) + j] = pixels[i][j]

            lbl = int.from_bytes(labels_file.read(1), byteorder='big')
            etiquetas_prueba[di] = lbl
            
    return valores_de_prueba, etiquetas_prueba
    
# Función para calcular la precisión
def calcular_accuracy(red, valores_prueba, etiquetas_prueba):
    total_ejemplos = len(valores_prueba)
    predicciones_correctas = 0

    for i in range(total_ejemplos):
        entrada = valores_prueba[i]
        #print("Entrada: ",len(entrada))
        etiqueta_real = int(etiquetas_prueba[i])
        #print("Etiqueta: ",etiqueta_real)
        salida_red = red.feed_forward(entrada)
        
        etiqueta_predicha = obtener_etiqueta_predicha(salida_red)
        
        if etiqueta_predicha == etiqueta_real:
            predicciones_correctas += 1
        
    accuracy = (predicciones_correctas / total_ejemplos)
    return accuracy

# Función para obtener la etiqueta predicha
def obtener_etiqueta_predicha(salida_red):
    etiqueta_predicha = np.argmax(salida_red)
    return etiqueta_predicha

# Función para entrenar una época
def entrenar_una_epoca(red, valores_entrenamiento, etiquetas_entrenamiento, num_datos,current_epoch):
    batch_size = sizeBatch
    # Registra el tiempo de inicio
    start_time = time.time()
    
    for i in range(0, num_datos, batch_size):
        entrada = []
        esperado = []
        
        for j in range(batch_size):
            dataIndex = i + j

            if dataIndex >= num_datos:
                break

            entrada.append(valores_entrenamiento[dataIndex])
            
            lab = np.zeros(10)
            lab[int(etiquetas_entrenamiento[dataIndex])] = 1
            
            esperado.append(lab)

            red.feed_forward(entrada[j])
            red.back_prop(esperado[j])

        red.update_network()
        
    # Registra el tiempo de finalización
    end_time = time.time()    
    # Calcula el tiempo transcurrido
    elapsed_time = end_time - start_time
    probar(red, valores_prueba, etiquetas_prueba,elapsed_time,current_epoch)

# Función para evaluar la red en un ejemplo de prueba
def probar(red, valores_prueba, etiquetas_prueba,elapsed_time,current_epoch):
    alazar = random.randint(0, len(valores_prueba) - 1)
    valores = red.feed_forward(valores_prueba[alazar])
    accuracy = calcular_accuracy(red, valores_prueba, etiquetas_prueba)
    print("Época: ",current_epoch+1," de ", num_epochs,"-",f"Modelo entrenado: {accuracy * 100:.2f}%","- Tiempo: ",round(elapsed_time)," segundos")
    # Crear y escribir en el archivo
    with open(nombre_archivo, "a") as archivo:
        info = f"Época: {current_epoch + 1} de {num_epochs} - Modelo entrenado: {accuracy * 100:.2f}% - Tiempo: {round(elapsed_time)} segundos\n"
        archivo.write(info)

num_datos_entrenamiento = 60000
num_datos_prueba = 10000

valores_entrenamiento, etiquetas_entrenamiento = inicializar("train-labels-idx1-ubyte.bytes", "train-images-idx3-ubyte.bytes", num_datos_entrenamiento)
valores_prueba, etiquetas_prueba = inicializarPrueba("t10k-labels-idx1-ubyte.bytes", "t10k-images-idx3-ubyte.bytes", num_datos_prueba)

red = NeuralNetwork([784, neurona_start, neurona_intermedia, 10], learning_rate)
#red= NeuralNetwork.load_network('mynetwork.bytes')
print("Empiezo")

nombre_archivo = f"{sizeBatch}_{num_epochs}_{neurona_start}_{neurona_intermedia}_{learning_rate}.txt"


for current_epoch in range(num_epochs):
    entrenar_una_epoca(red, valores_entrenamiento, etiquetas_entrenamiento, num_datos_entrenamiento,current_epoch)

red.save_network('mynetwork.bytes')    
print(f'Se ha creado el archivo "{nombre_archivo}" con la información proporcionada.')