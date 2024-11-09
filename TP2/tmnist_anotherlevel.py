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
# Leemos el archivo csv en un dataframe
df = pd.read_csv('TMNIST_Data.csv')

# Sacamos el atributo 'names'
df = df.drop('names', axis=1)

#%% Funciones

def imagen(df, digito):
    # Filtrar el DataFrame para obtener solo las filas que corresponden al dígito especificado
    imagenes_digito = df[df['labels'] == digito]
    
    # Seleccionar una imagen aleatoria del subconjunto filtrado
    imagen_seleccionada = imagenes_digito.sample(n=1).iloc[0, 1:].values
    
    # Convertir los valores de la imagen a un array de 8 bits y reformatear a 28x28
    pixels = np.array(imagen_seleccionada, dtype='uint8').reshape((28, 28))
    
    # Graficar la imagen
    plt.title(f'Dígito {digito}')
    plt.imshow(pixels, cmap='gray_r')  # Use 'gray_r' to invert the colors
    plt.show()
    
def mapacalornumeros(df):
    # Sumamos todas las filas para cada píxel
    pixel_sums = df.drop('labels', axis=1).sum(axis=0).values

    # Normalizamos las sumas de los píxeles
    pixel_sums_normalized = pixel_sums / pixel_sums.max()

    # Redimensionamos las sumas normalizadas de los píxeles en un array de 28x28
    heatmap_data = pixel_sums_normalized.reshape((28, 28))

    # Graficamos el mapa de calor
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig("figura1.png", dpi=300, bbox_inches='tight')
    plt.show()



#%% Codigo 

#%% Exploracion de datos
print(df.head())
print(df.info())
print('filas y columnas: ', df.shape)
print('tipos de datos: ', df.dtypes)
print('cantidad de digitos: ', df['labels'].value_counts())
print(df.describe())


#%% Sacar pixeles que suman cero

df_completo = df.copy()

pixel_sums = df.drop('labels', axis=1).sum(axis=0).values

#Cantidad de pixeles que suman 0
zero_sum_pixels = np.sum(pixel_sums == 0)
print(f'Cantidad de píxeles que sumaron 0: {zero_sum_pixels}')

df_labels = df['labels']

# columnas que no sumaron 0
non_zero_sum_pixels = np.sum(pixel_sums != 0)
print(f'Cantidad de píxeles que no sumaron 0: {non_zero_sum_pixels}')

#columnas que sumaron 0
cols_zero_sum = np.where(pixel_sums == 0)[0]
print(f'Columnas que sumaron 0: {cols_zero_sum.shape}')

# sacar columnas que sumaron 0
df = df.drop(df.columns[cols_zero_sum], axis=1)
print(f'Cantidad de columnas restantes: {df.shape[1]}')

df['labels'] = df_labels
# poner labels al principio
df = df[['labels'] + [col for col in df.columns if col != 'labels']]

df.head()

#%% Heatmap de presencia de pixeles
mapacalornumeros(df_completo)


#%% Imagenes digitos 1, 3 y 8 
imagen(df_completo, 1)
imagen(df_completo, 3)
imagen(df_completo, 8)

# Calculamos los valores promedio de los píxeles para cada etiqueta
promedios_labels = df_completo.groupby('labels').mean()

# Creamos una lista vacía para almacenar las diferencias de píxeles
pixel_diffs = []

recorridas = []

# Iteramos sobre todos los pares de etiquetas
for label1 in promedios_labels.index:
    for label2 in promedios_labels.index:
        # Nos aseguramos de no comparar una etiqueta consigo misma ni repetir comparaciones
        if label1 != label2 and (label2, label1) not in recorridas and (label1, label2) not in recorridas:
            # Calculamos la diferencia absoluta entre los valores promedio de los píxeles
            pixel_diff = (np.abs(promedios_labels.loc[label1] - promedios_labels.loc[label2])).mean()

            # Agregamos las diferencias de píxeles a la lista
            pixel_diffs.append((label1, label2, pixel_diff))

            # Registramos el par de etiquetas como ya comparado
            recorridas.append((label1, label2))

# Top 5 pares de labels con mayor diferencia promedio
top_5_differences = sorted(pixel_diffs, key=lambda x: x[2], reverse=True)[:5]

# Top 5 pares de labels con menor diferencia promedio
bottom_5_differences = sorted(pixel_diffs, key=lambda x: x[2])[:5]

print('Top 5 pares de labels con mayor diferencia promedio:')
for label1, label2, diff in top_5_differences:
    print(f'Labels {label1} y {label2}: {diff}')

print('\nTop 5 pares de labels con menor diferencia promedio:')
for label1, label2, diff in bottom_5_differences:
    print(f'Labels {label1} y {label2}: {diff}')
    
# Calculamos la diferencia absoluta entre el promedio de los pixeles de los labels 1 y 3
pixel_diff_1_3 = np.abs(promedios_labels.loc[1] - promedios_labels.loc[3])

# Normalizamos
pixel_diff_normalized_1_3 = pixel_diff_1_3 / pixel_diff_1_3.max()

# Hacemos reshape
heatmap_data_1_3 = pixel_diff_normalized_1_3.values.reshape((28, 28))

# Graficamos el heatmap
plt.imshow(heatmap_data_1_3, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

# Calculamos la diferencia absoluta entre el promedio de los pixeles de los labels 3 y 8
pixel_diff_3_8 = np.abs(promedios_labels.loc[3] - promedios_labels.loc[8])

# Normalizamos
pixel_diff_normalized_3_8 = pixel_diff_3_8 / pixel_diff_3_8.max()

# Hacemos reshape
heatmap_data_3_8 = pixel_diff_normalized_3_8.values.reshape((28, 28))

# Graficamos el heatmap
plt.imshow(heatmap_data_3_8, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

#%% Mapa de calor digito 1 con 3 y 3 con 8
# Nos quedamos con los digitos 1 y 3
df_1_3 = df_completo[(df_completo['labels'] == 1) | (df_completo['labels'] == 3)]

mapacalornumeros(df_1_3)

# Nos quedamos con los digitos 3 y 8
df_3_8 = df_completo[(df_completo['labels'] == 3) | (df_completo['labels'] == 8)]

mapacalornumeros(df_3_8)


#%% Digito 0 
for i in range(1, 5, 1):
    imagen(df_completo, 0)

# Mapa de calor 
# Nos quedamos el digito 0
df_0 = df_completo[(df_completo['labels'] == 0)]

mapacalornumeros(df_0)


#%% 