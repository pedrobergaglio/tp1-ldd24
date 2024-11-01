# Grupo: Another Level.
# Participantes: Pedro Bergaglio, Tomas Da Silva Minas y Maria Delfina Kiss. 
# Contenido: 
    
#%% Importaciones
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from inline_sql import sql

#%% Carga de Datos 
tmnist = pd.read_csv('/home/Estudiante/Descargas/TMNIST_Data.csv')

#%% Funciones


#%% Codigo 


#%% Exploracion de datos
print(tmnist.head())
print(tmnist.info())
print('filas y columnas: ', tmnist.shape)
print('tipos de datos: ', tmnist.dtypes)
print('cantidad de digitos: ', tmnist['labels'].value_counts())
print(tmnist.describe())


# Iterate over the DataFrame rows
for index, row in tmnist.iterrows():
    # The first column is the name
    nombre = row[0]
    label = row[1]

    # The rest of columns are pixels
    pixels = row[2:].values

    # Make those columns into a array of 8-bits pixels
    # This array will be of 1D with length 784
    # The pixel intensity values are integers from 0 to 255
    pixels = np.array(pixels, dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))

    # Plot
    plt.title('{nombre}:{label}'.format(nombre=nombre, label=label))
    plt.imshow(pixels, cmap='gray_r')  # Use 'gray_r' to invert the colors
    plt.show()

    break # Solo vemos la primer imagen
#%%
#digito 1
digito0 = tmnist[tmnist['labels'] == 1]

# Iterate over the DataFrame rows
k = 0
for index, row in digito0.iterrows():
    # The first column is the name
    nombre = row[0]
    label = row[1]

    # The rest of columns are pixels
    pixels = row[2:].values

    # Make those columns into a array of 8-bits pixels
    # This array will be of 1D with length 784
    # The pixel intensity values are integers from 0 to 255
    pixels = np.array(pixels, dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))

    # Plot
    plt.title('{nombre}:{label}'.format(nombre=nombre, label=label))
    plt.imshow(pixels, cmap='gray_r')  # Use 'gray_r' to invert the colors
    plt.show()
    
    k+=1
    
    if k == 6:
        break # Solo vemos la primer imagen
        
#%%
#digito 3        
        
digito0 = tmnist[tmnist['labels'] == 3]

# Iterate over the DataFrame rows
k = 0
for index, row in digito0.iterrows():
    # The first column is the name
    nombre = row[0]
    label = row[1]

    # The rest of columns are pixels
    pixels = row[2:].values

    # Make those columns into a array of 8-bits pixels
    # This array will be of 1D with length 784digito0 = tmnist[tmnist['labels'] == 3]

    # Iterate over the DataFrame rows
    k = 0
    for index, row in digito0.iterrows():
        # The first column is the name
        nombre = row[0]
        label = row[1]

        # The rest of columns are pixels
        pixels = row[2:].values

        # Make those columns into a array of 8-bits pixels
        # This array will be of 1D with length 784
        # The pixel intensity values are integers from 0 to 255
        pixels = np.array(pixels, dtype='uint8')

        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))

        # Plot
        plt.title('{nombre}:{label}'.format(nombre=nombre, label=label))
        plt.imshow(pixels, cmap='gray_r')  # Use 'gray_r' to invert the colors
        plt.show()
        
        k+=1
        
        if k == 6:
            break # Solo vemos la primer imagen
    # The pixel intensity values are integers from 0 to 255
    pixels = np.array(pixels, dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))

    # Plot
    plt.title('{nombre}:{label}'.format(nombre=nombre, label=label))
    plt.imshow(pixels, cmap='gray_r')  # Use 'gray_r' to invert the colors
    plt.show()
    
    k+=1
    
    if k == 6:
        break # Solo vemos la primer imagen
        
#%%

# Digito 8
digito0 = tmnist[tmnist['labels'] == 8]

# Iterate over the DataFrame rows
k = 0
for index, row in digito0.iterrows():
    # The first column is the name
    nombre = row[0]
    label = row[1]

    # The rest of columns are pixels
    pixels = row[2:].values

    # Make those columns into a array of 8-bits pixels
    # This array will be of 1D with length 784
    # The pixel intensity values are integers from 0 to 255
    pixels = np.array(pixels, dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))

    # Plot
    plt.title('{nombre}:{label}'.format(nombre=nombre, label=label))
    plt.imshow(pixels, cmap='gray_r')  # Use 'gray_r' to invert the colors
    plt.show()
    
    k+=1
    
    if k == 6:
        break # Solo vemos la primer imagen
        
#%%

digitos_1_y_0= sql^"""
SELECT *
FROM tmnist
WHERE labels=0 OR labels=1
"""