# Grupo: Another Level.
# Participantes: Pedro Bergaglio, Tomas Da Silva Minas y Maria Delfina Kiss. 
# Contenido: 
    
#%% Importaciones
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from inline_sql import sql
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict
from sklearn.metrics import confusion_matrix

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

#%% Valor absoluto diferencia pixeles
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

# Convertimos pixel_diffs a un DataFrame
df_diffs = pd.DataFrame(pixel_diffs, columns=['label1', 'label2', 'pixel_diff'])

# Creamos un scatter plot
plt.figure(figsize=(8, 6))

# Utilizamos el tamaño de los puntos para reflejar la diferencia de píxeles
plt.scatter(df_diffs['label1'], df_diffs['label2'], s=df_diffs['pixel_diff']*10, c=df_diffs['pixel_diff'], cmap='viridis', alpha=0.7)

# Agregar etiquetas y título
plt.colorbar(label='Diferencia absoluta de pixeles')
plt.xlabel('Digito 1')
plt.ylabel('Digito 2')
plt.grid(True)

# Mostrar el gráfico
plt.show()

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
    

#%% Imagenes digitos 1, 3 y 8 
imagen(df_completo, 1)
imagen(df_completo, 3)
imagen(df_completo, 8)

#%% Mapa de calor digito 1 con 3 y 3 con 8
# Nos quedamos con los digitos 1 y 3
df_1_3 = df_completo[(df_completo['labels'] == 1) | (df_completo['labels'] == 3)]

mapacalornumeros(df_1_3)

# Nos quedamos con los digitos 3 y 8
df_3_8 = df_completo[(df_completo['labels'] == 3) | (df_completo['labels'] == 8)]

mapacalornumeros(df_3_8)


#%% Diferencia absoluta digiros 1 con 3 y 3 con 8 
# Calculamos la diferencia absoluta entre el promedio de los pixeles de los labels 1 y 3
pixel_diff_1_3 = np.abs(promedios_labels.loc[1] - promedios_labels.loc[3])

# Normalizamos
pixel_diff_normalized_1_3 = pixel_diff_1_3 / pixel_diff_1_3.max()

# Hacemos reshape
data_1_3 = pixel_diff_normalized_1_3.values.reshape((28, 28))

# Graficar la imagen
plt.imshow(data_1_3, cmap='gray_r')  # Use 'gray_r' to invert the colors
plt.show()


# Calculamos la diferencia absoluta entre el promedio de los pixeles de los labels 3 y 8
pixel_diff_3_8 = np.abs(promedios_labels.loc[3] - promedios_labels.loc[8])

# Normalizamos
pixel_diff_normalized_3_8 = pixel_diff_3_8 / pixel_diff_3_8.max()

# Hacemos reshape
data_3_8 = pixel_diff_normalized_3_8.values.reshape((28, 28))

# Graficar la imagen
plt.imshow(data_3_8, cmap='gray_r')  # Use 'gray_r' to invert the colors
plt.show()


#%% Digito 0 
for i in range(1, 5, 1):
    imagen(df_completo, 0)

# Mapa de calor 
# Nos quedamos el digito 0
df_0 = df_completo[(df_completo['labels'] == 0)]

mapacalornumeros(df_0)


#%% Dataframe digitos 0 y 1 y exploracion
# Dataframe con imágenes correspondientes a los dígitos 0 o 1
df_0_1 = df[(df['labels'] == 0) | (df['labels'] == 1)]

# Cuántas muestras tenemos y si está balanceado
count_0 = df_0_1[df_0_1['labels'] == 0].shape[0]
count_1 = df_0_1[df_0_1['labels'] == 1].shape[0]
print(f'Cantidad de muestras del dígito 0: {count_0}')
print(f'Cantidad de muestras del dígito 1: {count_1}')


#%% Separamos los datos en conjuntos de train y de test
X_0_1 = df_0_1.drop('labels', axis=1).values
y_0_1 = df_0_1['labels'].values
X_train, X_test, y_train, y_test = train_test_split(X_0_1, y_0_1, test_size=0.2, random_state=42)


#%% 3 atributos a mano
#lista de 0 a 783
all_columns = np.arange(784)
#reshape a cuadrada
all_columns = all_columns.reshape((28, 28))

# Abajo a la izquierda
bottom_left = all_columns[20, 10]+1 # le agregamos uno porque arranca en 1
# Arriba a la derecha
top_right = all_columns[5, 20]+1
# Medio
middle = all_columns[14, 14]+1

atributo_elegido_mano = [str(bottom_left), str(top_right), str(middle)]


# Crear una matriz de 28x28 inicializada en cero
image_matrix = np.zeros((28, 28))

# Los índices seleccionados a mano
bottom_left = (20, 10)
top_right = (5, 20)
middle = (14, 14)

# Asignar intensidad 256 a los puntos seleccionados
image_matrix[bottom_left] = 256
image_matrix[top_right] = 256
image_matrix[middle] = 256

# Graficar la imagen
plt.imshow(image_matrix, cmap='gray_r', vmin=0, vmax=256)


X_mano = df_0_1[atributo_elegido_mano].values
y_mano = df_0_1['labels'].values

X_train_mano, X_test_mano, y_train_mano, y_test_mano = train_test_split(X_mano, y_mano, test_size=0.2, random_state=42)

knn_mano = KNeighborsClassifier(n_neighbors=3)
knn_mano.fit(X_train_mano, y_train_mano)
y_pred_mano = knn_mano.predict(X_test_mano)

accuracy_mano = accuracy_score(y_test_mano, y_pred_mano)
print(f'Exactitud con 3 los atributos elegidos a mano: {accuracy_mano}')


#%% 3 atributos con mayor diferencia absoluta
# Dataframe con imágenes correspondientes a los dígitos 0 o 1
df_1 = df[df['labels'] == 1]
df_0 = df[df['labels'] == 0]

# Calcuar el promedio para cada pixel
mean_1 = df_1.mean()
mean_0 = df_0.mean()

# Calcular la diferencia absoluta
diff = np.abs(mean_1 - mean_0)

# Calcular los 3 pixeles con mayor diferencia
diff_sorted = diff.sort_values(ascending=False)
atributo_elegido = diff_sorted.index[:3].values


# Crear una matriz de 28x28 inicializada en cero
image_matrix = np.zeros((28, 28))

# Asignar intensidad 256 a los píxeles seleccionados
for atributo in atributo_elegido:
    # Convertir el índice del atributo a su posición en la matriz 28x28
    row = int(atributo) // 28
    col = int(atributo) % 28
    image_matrix[row, col] = 256

# Graficar la imagen
plt.imshow(image_matrix, cmap='gray_r', vmin=0, vmax=256)


X_dif = df_0_1[atributo_elegido].values
y_dif = df_0_1['labels'].values

X_train_dif, X_test_dif, y_train_dif, y_test_dif = train_test_split(X_dif, y_dif, test_size=0.2, random_state=42)

knn_dif = KNeighborsClassifier(n_neighbors=3)
knn_dif.fit(X_train_dif, y_train_dif)
y_pred_dif = knn_dif.predict(X_test_dif)

accuracy_dif = accuracy_score(y_test_dif, y_pred_dif)
print(f'Exactitud con 3 los atributos según máxima diferencia absoluta entre pixeles: {accuracy_dif}')


#%% 3 atributos random
# 50 conjuntos de 3 atributos random distintos
sets_atributos = np.random.choice(range(1, 530), (50, 3), replace=False).tolist()

knn_rand = KNeighborsClassifier(n_neighbors=3)
preds = []

for attributes in sets_atributos:
    knn_rand.fit(X_train[:, attributes], y_train)
    y_pred_rand = knn_rand.predict(X_test[:, attributes])
    accuracy_rand = accuracy_score(y_test, y_pred_rand)
    preds.append((attributes, accuracy_rand))

# Atributos con mejor exactitud
best_attributes = max(preds, key=lambda x: x[1])[0]
best_accuracy = max(preds, key=lambda x: x[1])[1]
print(f'Mejores atributos: {best_attributes}, Exactitud: {best_accuracy}')


# Graficar los mejores atributos como una imagen de 28x28
image_matrix = np.zeros((28, 28))

# Asignar intensidad 256 a los píxeles correspondientes a los mejores atributos
for atributo in best_attributes:
    # Convertir el índice del atributo a su posición en la matriz 28x28
    row = atributo // 28
    col = atributo % 28
    image_matrix[row, col] = 256

# Graficar la imagen
plt.imshow(image_matrix, cmap='gray_r', vmin=0, vmax=256)


'''X_train_mejor = X_train[:, best_attributes]
X_test_mejor = X_test[:, best_attributes]

knn_mejor = KNeighborsClassifier(n_neighbors=3)
knn_mejor.fit(X_train_mejor, y_train)
y_pred_mejor = knn_mejor.predict(X_test_mejor)

accuracy_mejor = accuracy_score(y_test, y_pred_mejor)
print(f'Exactitud con 3 los atributos random con mejor exactitud: {accuracy_mejor}')'''


#%% Entrenar con diferentes cantidades de atributos
sets_atributos = np.random.choice(range(1, 530), (50, 3), replace=False).tolist()
sets_atributos += np.random.choice(range(1, 531), (50, 5), replace=False).tolist()
sets_atributos += np.random.choice(range(1, 531), (50, 7), replace=False).tolist()
sets_atributos += np.random.choice(range(1, 531), (50, 9), replace=False).tolist()
sets_atributos += np.random.choice(range(1, 531), (50, 11), replace=True).tolist()

knn_atributos = KNeighborsClassifier(n_neighbors=3)
preds = []

for attributes in sets_atributos:
    knn_atributos.fit(X_train[:, attributes], y_train)
    y_pred = knn_atributos.predict(X_test[:, attributes])
    accuracy = accuracy_score(y_test, y_pred)
    preds.append((attributes, accuracy))

# Atributos con mejor exactitud
best_attributes = max(preds, key=lambda x: x[1])[0]

# Diccionario para almacenar las exactitudes según la cantidad de atributos
accuracy_by_num_attributes = defaultdict(list)

# Calcular la exactitud promedio según la cantidad de atributos
for attributes, accuracy in preds:
    num_attributes = len(attributes)
    accuracy_by_num_attributes[num_attributes].append(accuracy)

# Calcular el promedio de exactitud para cada cantidad de atributos
average_accuracy_by_num_attributes = {num_attributes: np.mean(accuracies) for num_attributes, accuracies in accuracy_by_num_attributes.items()}

# Imprimir los resultados
for num_attributes, avg_accuracy in average_accuracy_by_num_attributes.items():
    print(f'Cantidad de atributos: {num_attributes}, Exactitud promedio: {avg_accuracy}')
    
#%% Grafico para mostrar variacion de la exactitud por cantidad de atributos
# Extraer los datos para el gráfico
num_attributes_list = sorted(average_accuracy_by_num_attributes.keys())
accuracy_list = [average_accuracy_by_num_attributes[num] for num in num_attributes_list]

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(num_attributes_list, accuracy_list, marker='o', linestyle='-', color='r')

# Configurar etiquetas y títulos
plt.xlabel('Cantidad de Atributos')
plt.ylabel('Exactitud Promedio')
plt.ylim(0.7, 1.0)
plt.xticks(num_attributes_list)  # Asegura que todos los números de atributos aparezcan en el eje x

# Mostrar los valores exactos en cada punto
for i, acc in enumerate(accuracy_list):
    plt.text(num_attributes_list[i], acc - 0.015, f'{acc:.4f}', ha='center', va='top')

# Agregar cuadrícula y leyenda
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()
    

#%% Variacion de cantidad de atributos y de vecinos
k_values = [1, 3, 5, 10, 20, 50]
best_accuracy = 0
best_k = 0
best_attributes = []

results = []

for k in k_values:
    for attributes in sets_atributos:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train[:, attributes], y_train)
        y_pred = knn.predict(X_test[:, attributes])
        accuracy = accuracy_score(y_test, y_pred)

        results.append((k, attributes, accuracy))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_attributes = attributes
            precision = np.mean(y_pred == y_test)
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)

print(f'Mejor modelo - k: {best_k}, Atributos: {best_attributes}, Exactitud: {best_accuracy}')
print(f'Precisión: {precision}')
print(f'Matriz de confusión:\n{cm}')

#%% Graficamos
# Calcular la exactitud promedio para cada k
accuracy_by_k = defaultdict(list)

for k, attributes, accuracy in results:
    accuracy_by_k[k].append((attributes, accuracy))

average_accuracy_by_k_attributes = {}

for k, attributes_accuracies in accuracy_by_k.items():
    for attributes, accuracies in attributes_accuracies:
        average_accuracy_by_k_attributes[f'{k}, {len(attributes)} attributes'] = np.mean(accuracies)

for k_attributes, avg_accuracy in average_accuracy_by_k_attributes.items():
    print(f'{k_attributes}: {avg_accuracy}')    
    
# Graficamos
# Extraer los valores de k, cantidad de atributos y exactitud promedio
k_values = [int(key.split(',')[0]) for key in average_accuracy_by_k_attributes.keys()]
num_attributes = [int(key.split(',')[1].split(' ')[1]) for key in average_accuracy_by_k_attributes.keys()]
accuracies = list(average_accuracy_by_k_attributes.values())

# Crear un scatter plot
scatter = plt.scatter(k_values, accuracies, c=num_attributes, cmap='viridis')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Exactitud promedio')
# Agregar la barra de color
cbar = plt.colorbar(scatter)
cbar.set_label('Cantidad de Atributos')
plt.show()
    
