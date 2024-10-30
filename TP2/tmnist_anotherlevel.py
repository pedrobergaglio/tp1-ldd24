# Grupo: Another Level.
# Participantes: Pedro Bergaglio, Tomas Da Silva Minas y Maria Delfina Kiss. 
# Contenido: 
    
#%% Importaciones
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% Carga de Datos 
tmnist = pd.read_csv('TMNIST_Data.csv')

#%% Funciones


#%% Codigo 


#%% Exploracion de datos
print(tmnist.head())
print(tmnist.info())
print(tmnist.shape)

print(tmnist.dtypes)
# Convertimos las columnas de píxeles a tipo float, ignorando posibles errores
tmnist.iloc[:, 1:] = tmnist.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

print(tmnist['labels'].value_counts())
print(tmnist.describe())
sns.countplot(x='labels', data=tmnist)



img_data = tmnist.iloc[1, 2:].values  # Saltar la columna de etiqueta

# Convertir a una matriz de 28x28 y graficar
img = img_data.reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.show()
