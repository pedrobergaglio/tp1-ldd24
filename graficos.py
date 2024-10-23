import pandas as pd
from inline_sql import sql, sql_val
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as ticker
from matplotlib import rcParams
import seaborn as sns

paises = pd.read_csv('/home/delfikiss/Desktop/Laboratorio de Datos/TP1/tp1-ldd24-main/exports/paises.csv')
migracion = pd.read_csv('/home/delfikiss/Desktop/Laboratorio de Datos/TP1/tp1-ldd24-main/exports/emigracion.csv')
sedes = pd.read_csv('/home/delfikiss/Desktop/Laboratorio de Datos/TP1/tp1-ldd24-main/exports/sedes.csv')
redes = pd.read_csv('/home/delfikiss/Desktop/Laboratorio de Datos/TP1/tp1-ldd24-main/exports/redes_sociales.csv')

#%% i)
# Calculamos la cantidad de sedes por region geografica usando consultas de SQL
sede_region = sql^ """ 
            SELECT s.sede_id, p.region_geografica
            FROM sedes AS s
            INNER JOIN paises AS p
            ON s.ISO3=p.ISO3;
"""

cantidad_sedes = """
                SELECT region_geografica AS 'Region Geografica', COUNT(sede_id) AS 'Cantidad de Sedes'
                FROM sede_region
                GROUP BY region_geografica
                ORDER BY count(sede_id) DESC;
"""

ejercicio_i = sql^ cantidad_sedes


plt.figure(figsize=(8, 5))
# horizontal
ax = sns.barplot(x='Cantidad de Sedes', y='Region Geografica', data=ejercicio_i, color='seagreen')
# numeros al final de las barras 
for index, value in enumerate(ejercicio_i['Cantidad de Sedes']):
    ax.text(value + 0.3, index, f'{value}', va='center', ha='left', color='black')
ax.set_title('Cantidad de Sedes por Region Geografica', fontsize=14, weight='bold')
ax.set_xticks([])

# Exportamos el grafico a la carpeta 'graficos'
plt.savefig('graficos/grafico_i.png', bbox_inches='tight')

# Mostramos el grafico
plt.show()


#%% ii)
# Calculamos para cada region geografica el promedio del flujo migratorio de los paises donde argentina tiene una delegacion usando pandas

# Filtrar migraciones para los años 1960, 1970, 1980, 1990, y 2000
migraciones = migracion[migracion['anio'].isin([1960, 1970, 1980, 1990, 2000])]

# Unir las sedes con las migraciones (tanto como origen y destino)
sedes_origen = pd.merge(sedes, migraciones, left_on='ISO3', right_on='ISO3_origen')
sedes_destino = pd.merge(sedes, migraciones, left_on='ISO3', right_on='ISO3_destino')

# Agrupar por ISO3 para obtener la suma del flujo migratorio
flujo_origen = sedes_origen.groupby('ISO3')['cantidad'].sum().reset_index().rename(columns={'cantidad': 'emigracion'})
flujo_destino = sedes_destino.groupby('ISO3')['cantidad'].sum().reset_index().rename(columns={'cantidad': 'inmigracion'})

# Unir las emigraciones e inmigraciones
flujo_total = pd.merge(flujo_origen, flujo_destino, on='ISO3', how='outer').fillna(0)

# Calcular el flujo migratorio promedio (la diferencia entre inmigración y emigración, dividido entre 5 años)
flujo_total['flujo_migratorio'] = (flujo_total['inmigracion'] - flujo_total['emigracion']) / 5

# Unir con la tabla de países para obtener la región geográfica
flujo_paises_region = pd.merge(flujo_total, paises[['ISO3', 'region_geografica']], on='ISO3')

# Calcular la mediana del flujo migratorio por región geográfica
medianas = flujo_paises_region.groupby('region_geografica')['flujo_migratorio'].median().sort_values(ascending=False)


# Crear la figura
plt.figure(figsize=(10, 20))

# Graficar el boxplot, ordenado por la mediana de cada región
sns.boxplot(x='region_geografica', y='flujo_migratorio', data=flujo_paises_region, 
            showmeans=True, order=medianas.index, color='skyblue')

# Agregar título y etiquetas a los ejes
plt.title('Promedio del Flujo Migratorio por Región Geográfica')
plt.xlabel('Región Geográfica')
plt.ylabel('Flujo Migratorio')

# Rotar etiquetas en el eje x
plt.xticks(rotation=90, ha="right")

# Limitar el rango de valores de los ejes automáticamente
q1 = flujo_paises_region['flujo_migratorio'].quantile(0.25)
q3 = flujo_paises_region['flujo_migratorio'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 30 * iqr
upper_bound = q3 + 130 * iqr
plt.ylim(lower_bound, upper_bound)

# Exportamos el grafico a la carpeta 'graficos'
plt.savefig('graficos/grafico_ii_a.png', bbox_inches='tight')

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# quitar AMERICA DEL NORTE
flujo_paises_region = flujo_paises_region[flujo_paises_region['region_geografica'] != 'AMÉRICA  DEL  NORTE']

# Calcular la mediana del flujo migratorio por región geográfica
medianas = flujo_paises_region.groupby('region_geografica')['flujo_migratorio'].median().sort_values(ascending=False)

# Crear la figura
plt.figure(figsize=(10, 20))

# Graficar el boxplot, ordenado por la mediana de cada región
sns.boxplot(x='region_geografica', y='flujo_migratorio', data=flujo_paises_region, 
            showmeans=True, order=medianas.index, color='skyblue')

# Agregar título y etiquetas a los ejes
plt.title('Promedio del Flujo Migratorio por Región Geográfica')
plt.xlabel('Región Geográfica')
plt.ylabel('Flujo Migratorio')

# Rotar etiquetas en el eje x
plt.xticks(rotation=90, ha="right")

# Limitar el rango de valores de los ejes automáticamente
q1 = flujo_paises_region['flujo_migratorio'].quantile(0.25)
q3 = flujo_paises_region['flujo_migratorio'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 30 * iqr
upper_bound = q3 + 130 * iqr
plt.ylim(lower_bound, upper_bound)

# Exportamos el grafico a la carpeta 'graficos'
plt.savefig('graficos/grafico_ii_b.png', bbox_inches='tight')

# Mostrar el gráfico
plt.tight_layout()
plt.show()


#%% 
# Realizamos una consulta de SQL para obtener los flujo migratorio de los paises hacia argentina en el año 2000 y cuantas sedes argentinas tiene ese pais
inmigrantes_arg = sql^ """
                SELECT ISO3_origen AS ISO3, cantidad AS flujo_migratorio
                FROM migracion
                WHERE anio='2000' AND ISO3_destino='ARG';
"""

cantidad_sedes = sql^ """
                SELECT ISO3, COUNT(sede_id) AS cant_sedes
                FROM sedes
                GROUP BY ISO3;
"""

flujo_sedes = """
            SELECT cs.ISO3, ia.flujo_migratorio AS 'Flujo Migratorio', cs.cant_sedes AS 'Cantidad de Sedes'
            FROM inmigrantes_arg AS ia
            INNER JOIN cantidad_sedes AS cs
            ON cs.ISO3 = ia.ISO3;
"""

ejercicio_iii = sql^ flujo_sedes


# Graficamos la relación con un scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Flujo Migratorio', y='Cantidad de Sedes', data=ejercicio_iii, color='red')

# Etiquetas y título
plt.xlabel('Cantidad de Migrantes hacia Argentina (año 2000)')
plt.ylabel('Cantidad de Sedes de Argentina en el Exterior')
plt.title('Relación entre Migrantes y Sedes en el Exterior')

# Exportamos el grafico a la carpeta 'graficos'
plt.savefig('graficos/grafico_iii.png', bbox_inches='tight')

# Mostramos el gráfico
plt.show()
