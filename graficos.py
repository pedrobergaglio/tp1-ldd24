#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:21:05 2024

@author: delfikiss
"""

import pandas as pd
from inline_sql import sql, sql_val
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as ticker
from matplotlib import rcParams
import seaborn as sns

pais = pd.read_csv('/home/delfikiss/Downloads/tp1-ldd24-main/exports/paises.csv')
emigracion = pd.read_csv('/home/delfikiss/Downloads/tp1-ldd24-main/exports/emigracionok.csv')
sedes = pd.read_csv('/home/delfikiss/Downloads/tp1-ldd24-main/exports/sedes.csv')
redes = pd.read_csv('/home/delfikiss/Downloads/tp1-ldd24-main/exports/redes_sociales.csv')

emigracion.rename(columns={'año': 'anio'}, inplace=True)

#%% i)
# Calculamos la cantidad de sedes por region geografica usando consultas de SQL
sede_region = sql^ """ 
            SELECT s.sede_id, p.region_geografica
            FROM sedes AS s
            INNER JOIN pais AS p
            ON s.ISO3=p.ISO3;
"""

cantidad_sedes = """
                SELECT region_geografica AS 'Region Geografica', COUNT(sede_id) AS 'Cantidad de Sedes'
                FROM sede_region
                GROUP BY region_geografica
                ORDER BY count(sede_id) DESC;
"""

ejercicio_i = sql^ cantidad_sedes

print(ejercicio_i)

# Generamos un grafico de barras para mostrar los resultados
# CON MATPLOT VERTICAL
fig, ax = plt.subplots()

plt.rcParams['font.family'] = 'sans-serif'           


ax.bar(data=ejercicio_i, x='Region Geografica', height='Cantidad de Sedes', color='green')
       
ax.set_title('Cantidad de Sedes por Region Geografica')
ax.set_xlabel('Region Geografica', fontsize='medium')                       
ax.set_ylabel('Cantidad de Sedes', fontsize='medium')    
ax.set_xlim(-1, 7)
ax.set_ylim(0, 35)

ax.bar_label(ax.containers[0], fontsize=8)   # Agrega la etiqueta a cada barra
plt.xticks(rotation=90, ha="right")

# CON SEABORN HORIZONTAL
plt.figure(figsize=(8, 5))
# horizontal
ax = sns.barplot(x='Cantidad de Sedes', y='Region Geografica', data=ejercicio_i, color='seagreen')
# numeros al final de las barras 
for index, value in enumerate(ejercicio_i['Cantidad de Sedes']):
    ax.text(value + 0.3, index, f'{value}', va='center', ha='left', color='black')
ax.set_title('Cantidad de Sedes por Region Geografica', fontsize=14, weight='bold')
ax.set_xticks([])
plt.show()

#%% ii)
# Calculamos para cada region geografica el promedio del flujo migratorio de los paises donde argentina tiene una delegacion
paises_delegacion = sql^ """
                SELECT ISO3_origen, ISO3_destino, anio, cantidad
                FROM emigracion
                INNER JOIN sedes
                ON ISO3=ISO3_origen OR ISO3=ISO3_destino;
"""

origen = sql^ """
            SELECT ISO3_origen, COUNT(CAST(cantidad AS INTEGER)) AS emigracion
            FROM paises_delegacion
            GROUP BY ISO3_origen;
"""

destino = sql^ """
            SELECT ISO3_destino, COUNT(CAST(cantidad AS INTEGER)) AS inmigracion
            FROM paises_delegacion
            GROUP BY ISO3_destino;
"""

flujo_pais = sql^ """
            SELECT ISO3_origen AS ISO3, (inmigracion-emigracion)/5 AS flujo_migratorio
            FROM origen
            INNER JOIN destino
            ON ISO3_origen=ISO3_destino;
"""

region_flujo = sql^"""
            SELECT p.region_geografica, f.flujo_migratorio
            FROM flujo_pais AS f
            INNER JOIN pais AS p
            ON f.ISO3=p.ISO3;
"""

rcParams['font.family'] = 'sans-serif'            # Modifica el tipo de letra
rcParams['axes.spines.right']  = False            # Elimina linea derecha   del recuadro
rcParams['axes.spines.left']   = True             # Agrega  linea izquierda del recuadro
rcParams['axes.spines.top']    = False            # Elimina linea superior  del recuadro
rcParams['axes.spines.bottom'] = False            # Elimina linea inferior  del recuadro

fig, ax = plt.subplots()

region_flujo.boxplot(by=['region_geografica'], column=['flujo_migratorio'], 
             ax=ax, grid=False, showmeans=True)

# Agrega titulo, etiquetas a los ejes  
fig.suptitle('')
ax.set_title('Promedio del Flujo Migratorio por Region Geografica')
ax.set_xlabel('Region Geografica')
ax.set_ylabel('Flujo Migratorio')

plt.xticks(rotation=90, ha="right")
ax.set_ylim(-10,12) #limita el rango de valores de los ejes