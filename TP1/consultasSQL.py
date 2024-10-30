# Integrantes: Bergaglio Pedro, Da Silva Minas Tomas y Kiss Maria Delfina
# En este archivo realizamos las respectivas consultas de SQL que utilizamos a lo largo del trabajo

import pandas as pd
from inline_sql import sql, sql_val
import matplotlib.pyplot as plt

paises = pd.read_csv('esquemas/paises.csv')
migracion = pd.read_csv('esquemas/migracion.csv')
sedes = pd.read_csv('esquemas/sedes.csv')
redes = pd.read_csv('esquemas/redes_sociales.csv')

#%% i)
# De la tabla sedes contamos la cantidad de sedes totales por cada pais y el promedio de la cantidad de secciones por pais, agrupamos por ISO3
sedes_secciones = sql^ """
                SELECT ISO3, COUNT(sede_id) AS sedes, AVG(cantidad_secciones) AS secciones_promedio
                FROM sedes
                GROUP BY ISO3;
"""
# De migracion sumamos la cantidad y elegimos solo los datos del año 2000, luego agrupamos por ISO3. 
# Nos quedamos con los ISO3_origen para saber que cantidad de personas emigro desde ese pais 
emigra = sql^ """
            SELECT ISO3_origen, SUM(CAST(cantidad AS INTEGER)) AS cantidad
            FROM migracion
            WHERE anio='2000'
            GROUP BY ISO3_origen;
            
"""
# De migracion sumamos la cantidad y elegimos solo los datos del año 2000, luego agrupamos por ISO3.
# Nos quedamos con los ISO3_destino para saber la cantidad de personas que inmigro a ese pais
inmigra = sql^ """
            SELECT ISO3_destino, SUM(CAST(cantidad AS INTEGER)) AS cantidad
            FROM migracion
            WHERE anio='2000'
            GROUP BY ISO3_destino;
"""
# Nos quedamos con ISO3_origen de la tabla emigra, y creamos el flujo neto restando cantidad de inmigra y cantidad de emigra
# Juntamos las tablas con un INNER JOIN usando ISO3 como referencia para el join
flujo_neto = sql^ """
            SELECT e.ISO3_origen AS ISO3, i.cantidad - e.cantidad AS neto
            FROM emigra AS e
            INNER JOIN inmigra AS i
            ON e.ISO3_origen=i.ISO3_destino;
"""
# Junta flujo_neto y sedes_secciones para poder agrupar, sedes, secciones_promedio, ISO3 y neto
sedes_flujo = sql^ """
            SELECT s.ISO3, s.sedes, s.secciones_promedio, fn.neto
            FROM sedes_secciones AS s
            INNER JOIN flujo_neto AS fn
            ON s.ISO3=fn.ISO3;
"""
# Para terminar juntamos sedes_flujo con paises para quedarnos con todos los atributos de sedes_flujos excepto ISO3 y ademas con el atributo nombre de paises
paises_flujo = """
            SELECT p.nombre AS Pais, sf.sedes, sf.secciones_promedio AS 'secciones promedio', sf.neto AS 'flujo migratorio neto'
            FROM sedes_flujo AS sf
            INNER JOIN paises AS p
            ON sf.ISO3=p.ISO3
            ORDER BY sf.sedes DESC, p.nombre ASC;
"""

consulta_i = sql^ paises_flujo


# Guardamos una imagen del head del data frame para el informe
head_i = consulta_i.head()

# Creamos una figura y un eje para la tabla
fig, ax = plt.subplots(figsize=(8, 2))  # Ajusta el tamaño de la imagen
ax.axis('tight')
ax.axis('off')

# Ajustamos el tamaño de las columnas basado en el contenido
# Calcular los anchos de las columnas según el contenido
col_widths = [max(len(str(col)) for col in head_i[col_name]) * 0.1 for col_name in head_i.columns]

# Dibujamos la tabla en el gráfico con los anchos de columna ajustados
table = ax.table(cellText=head_i.values, colLabels=head_i.columns, cellLoc='center', loc='center')

# Ajustamos los anchos de las columnas
for i, width in enumerate(col_widths):
    table.auto_set_column_width([i])  # Ajuste automático de las columnas
    table.scale(1, 1.5)  # Ajuste el tamaño de la tabla en general

# Ajustamos el ancho de filas
table.scale(1, 0.5)  # El primer valor escala el ancho, el segundo escala la altura de las filas

# Guardamos la tabla como PNG
plt.savefig('consultas/head_consulta_i.png', bbox_inches='tight', dpi=300)
plt.show()


# Exportamos el data frame a .csv en la carpeta 'consultas' 
consulta_i.to_csv('consultas/consulta_i.csv', index=False)


#%% ii)
# Juntamos sedes con paises para quedarnos el ISO3 de sedes y la region geografica de paises
paises_sedes = sql^"""
                    SELECT DISTINCT s.ISO3, region_geografica
                    FROM sedes AS s
                    INNER JOIN paises AS p
                    ON s.ISO3=p.ISO3;
"""
# Sumamos la cantidad de sedes por region geografica, agrupamos por region geografica
cantidad = sql^"""
                SELECT COUNT(ISO3) AS 'cant', region_geografica
                FROM paises_sedes 
                GROUP BY region_geografica;
"""
# Juntamos el atributo region geografica de paises con la cantidad de migracion para saber el flujo de emigracion de argentina 
flujo_emigracion = sql^ """
                    SELECT cantidad, region_geografica
                    FROM migracion
                    INNER JOIN paises
                    ON ISO3_destino = ISO3
                    WHERE ISO3_origen = 'ARG';
"""
# Hacemos el promedio de la cantidad de emigracion segun region geografica de flujo_emigracion
promedio_flujo = sql^ """
                    SELECT AVG(CAST(cantidad AS INTEGER)) as promedio, region_geografica
                    FROM flujo_emigracion
                    GROUP BY region_geografica;
"""
# Juntamos promedio_flujo y cantidad
# Nos quedamos con region geografica de cantidad, cant de cantidad y promedio de promedio_flujo 
pais_promedio = """
                SELECT c.region_geografica AS 'Region Geografica', c.cant AS 'Paises Con Sedes Argentinas', pf.promedio AS 'Promedio flujo con Argentina - Países con Sedes Argentinas'
                FROM cantidad AS c
                INNER JOIN promedio_flujo AS pf
                ON c.region_geografica=pf.region_geografica
                ORDER BY pf.promedio DESC;
"""

consulta_ii = sql^ pais_promedio


# Guardamos una imagen del head del data frame para el informe
head_ii = consulta_ii.head()

# Creamos una figura y un eje para la tabla
fig, ax = plt.subplots(figsize=(8, 2))  # Ajusta el tamaño de la imagen
ax.axis('tight')
ax.axis('off')

# Ajustamos el tamaño de las columnas basado en el contenido
# Calcular los anchos de las columnas según el contenido
col_widths = [max(len(str(col)) for col in head_ii[col_name]) * 0.1 for col_name in head_ii.columns]

# Dibujamos la tabla en el gráfico con los anchos de columna ajustados
table = ax.table(cellText=head_ii.values, colLabels=head_ii.columns, cellLoc='center', loc='center')

# Ajustamos los anchos de las columnas
for i, width in enumerate(col_widths):
    table.auto_set_column_width([i])  # Ajuste automático de las columnas
    table.scale(1, 1.5)  # Ajuste el tamaño de la tabla en general
    
# Ajustamos el ancho de filas
table.scale(1, 0.5)  # El primer valor escala el ancho, el segundo escala la altura de las filas

# Guardamos la tabla como PNG
plt.savefig('consultas/head_consulta_ii.png', bbox_inches='tight', dpi=300)
plt.show()


# Exportamos el data frame a .csv en la carpeta 'consultas' 
consulta_ii.to_csv('consultas/consulta_ii.csv', index=False)


#%% iii)
# INNER JOIN de sedes y redes con sede_id, para quedarnos con plataforma e ISO3 sin repetidos 
sedes_redes = sql^"""
            SELECT DISTINCT plataforma, ISO3
            FROM sedes AS s
            INNER JOIN redes AS r
            ON s.sede_id=r.sede_id;
"""
# Contamos la cantidad de paltaformas que tiene cada pais
redes_pais = sql^"""
            SELECT COUNT(plataforma) AS cantidad, ISO3
            FROM sedes_redes AS sr
            GROUP BY ISO3;
"""
# Juntamos paises con redes_pais para quedarnos con el pais y con la cantidad de redes por pais
cantidad_redes = """
                SELECT nombre AS Pais, cantidad AS 'Cantidad Redes'
                FROM redes_pais AS rp
                INNER JOIN paises AS p
                ON rp.ISO3=p.ISO3
                ORDER BY cantidad DESC;
"""

consulta_iii = sql^ cantidad_redes


# Guardamos una imagen del head del data frame para el informe
head_iii = consulta_iii.head()

# Creamos una figura y un eje para la tabla
fig, ax = plt.subplots(figsize=(8, 2))  # Ajusta el tamaño de la imagen
ax.axis('tight')
ax.axis('off')

# Ajustamos el tamaño de las columnas basado en el contenido
# Calcular los anchos de las columnas según el contenido
col_widths = [max(len(str(col)) for col in head_iii[col_name]) * 0.1 for col_name in head_iii.columns]

# Dibujamos la tabla en el gráfico con los anchos de columna ajustados
table = ax.table(cellText=head_iii.values, colLabels=head_iii.columns, cellLoc='center', loc='center')

# Ajustamos los anchos de las columnas
for i, width in enumerate(col_widths):
    table.auto_set_column_width([i])  # Ajuste automático de las columnas
    table.scale(1, 1.5)  # Ajuste el tamaño de la tabla en general

# Ajustamos el ancho de filas
table.scale(1, 0.75)  # El primer valor escala el ancho, el segundo escala la altura de las filas

# Guardamos la tabla como PNG
plt.savefig('consultas/head_consulta_iii.png', bbox_inches='tight', dpi=300)
plt.show()


# Exportamos el data frame a .csv en la carpeta 'consultas' 
consulta_iii.to_csv('consultas/consulta_iii.csv', index=False)


#%% iv)
# Juntamos redes con sedes por sede_id para quedarnos con el url y plataforma de redes y la sede_id de sedes
redes_sedes = sql^ """
               SELECT url, plataforma, r.sede_id, ISO3
               FROM redes AS r
               INNER JOIN sedes AS s
               ON r.sede_id=s.sede_id;
              """
# Juntamos paises y redes_sedes por ISO3 para poder mostrar por pais, sus sedes con sus plataformas y respectivos URL 
# Ordenamos por los atributos de manera ascendente
redes_sociales = """
                SELECT nombre AS Pais, sede_id AS Sede, plataforma AS 'Red Social', url AS URL
                FROM redes_sedes AS r
                INNER JOIN paises AS p
                ON r.ISO3=p.ISO3
                ORDER BY nombre ASC, sede_id  ASC, plataforma ASC, url ASC;
"""

consulta_iv = sql^ redes_sociales


# Guardamos una imagen del head del data frame para el informe
head_iv = consulta_iv.head()

# Creamos una figura y un eje para la tabla
fig, ax = plt.subplots(figsize=(8, 2))  # Ajusta el tamaño de la imagen
ax.axis('tight')
ax.axis('off')

# Ajustamos el tamaño de las columnas basado en el contenido
# Calcular los anchos de las columnas según el contenido
col_widths = [max(len(str(col)) for col in head_iv[col_name]) * 0.1 for col_name in head_iv.columns]

# Dibujamos la tabla en el gráfico con los anchos de columna ajustados
table = ax.table(cellText=head_iv.values, colLabels=head_iv.columns, cellLoc='center', loc='center')

# Ajustamos los anchos de las columnas
for i, width in enumerate(col_widths):
    table.auto_set_column_width([i])  # Ajuste automático de las columnas
    table.scale(1, 1.5)  # Ajuste el tamaño de la tabla en general

# Ajustamos el ancho de filas
table.scale(1, 0.5)  # El primer valor escala el ancho, el segundo escala la altura de las filas

# Guardamos la tabla como PNG
plt.savefig('consultas/head_consulta_iv.png', bbox_inches='tight', dpi=300)
plt.show()


# Exportamos el data frame a .csv en la carpeta 'consultas' 
consulta_iv.to_csv('consultas/consulta_iv.csv', index=False)
