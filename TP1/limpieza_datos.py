 
#importar librerias y leer los archivos csv
import pandas as pd

df_sedes = pd.read_csv('data original/lista-sedes.csv')
df_migraciones = pd.read_csv('data original/datos_migraciones.csv')
df_secciones = pd.read_csv('data original/lista-secciones.csv')
# hay una fila que falla, por lo que se ignoran los errores
df_sedes_datos = pd.read_csv('data original/lista-sedes-datos.csv', sep=',', error_bad_lines=False, warn_bad_lines=True)

 
# cantidad de veces que aparece el valor 'Argentinos  en  el  exterior' en la columna 'ciudad' de la tabla df_lista_sedes
df_sedes_ciudad = df_sedes[df_sedes['ciudad_castellano'] == 'Argentinos  en  el  exterior']
print('Cantidad de veces que aparece el valor "Argentinos  en  el  exterior" en la columna "ciudad_castellano" de la tabla df_lista_sedes: ', df_sedes_ciudad.shape[0])
# cantidad de sedes
print("Cantidad de sedes: ", df_sedes.shape[0])

# cantidad de valores en migraciones que tienen .. en la columna '2000 [2000]'
df_migraciones_2000 = df_migraciones[df_migraciones['2000 [2000]'] == '..']
print('Cantidad de valores en migraciones que tienen ".." en la columna "2000 [2000]": ', df_migraciones_2000.shape[0])
# cantidad total
print('Cantidad total de valores en migraciones: ', df_migraciones.shape[0])


 
# cantidad de valores que tienen caracteres no dígitos o - en la columna 'codigo_postal' en la tabla lista-sedes-datos.csv
df_sedes_datos_codigo_postal = df_sedes_datos[df_sedes_datos['codigo_postal'].str.contains(r'\D|-', na=False)]
print('Cantidad de valores que tienen caracteres no dígitos o - en la columna "codigo_postal" en la tabla lista-sedes-datos.csv: ', df_sedes_datos_codigo_postal.shape[0])
# cantidad de sedes
print("Cantidad de sedes: ", df_sedes.shape[0])

 
# cantidad de valores que no son nulos, y no tienen una dirección de correo válida (una sola palabra y tiene @) en la columna 'correo_electronico' en la tabla lista-secciones.csv
df_secciones_no_validas = df_secciones[
    df_secciones['correo_electronico'].notna() &
    (df_secciones['correo_electronico'].astype(str).str.contains(" ") |
    ~df_secciones['correo_electronico'].astype(str).str.contains("@"))
    ]
print('Cantidad de valores que tienen direcciones no válidas en la columna "correo_electronico" en la tabla lista-secciones.csv: ', df_secciones_no_validas.shape[0])
# cantidad de secciones
print("Cantidad de secciones: ", df_secciones.shape[0])

 
# ELIMINAR COLUMNAS QUE SABEMOS QUE NO VAMOS A USAR

# MIGRACIONES
# las columnas con nombres son rebundantes, las podemos eliminar
df_migraciones = df_migraciones.drop(columns=['Migration by Gender Code', 'Country Origin Name', 'Country Dest Name'])

# la columna 'Migration by Gender Code'
# no es necesaria para las consultas, basta con utilizar las filas que tienen Total
# filtramos las filas que tienen Total en 'Migration by Gender Name'
df_migraciones = df_migraciones[df_migraciones['Migration by Gender Name'] == 'Total']
# eliminamos la columna 'Migration by Gender Name'
df_migraciones = df_migraciones.drop(columns=['Migration by Gender Name'])


 
# valores unicos en sedes columna sede_tipo
print("Valores unicos en sedes columna sede_tipo:", df_sedes['sede_tipo'].unique())
print("Valores unicos en sedes columna estado:", df_sedes['estado'].unique())

 
# SEDES

# Las columnas sede_desc_ingles,pais_iso_2,pais_iso_3,pais_ingles,ciudad_ingles, no son necesarias para las consultas
df_sedes = df_sedes.drop(columns=['pais_iso_2', 'ciudad_ingles'])


 
# unique values column temas 
print("Valores de la columna 'temas':", df_secciones['temas'].unique())

 
# SECCIONES
# las columnas 'sede_desc_ingles' no es necesaria ya que ya tenemos el nombre en español
df_secciones = df_secciones.drop(columns=['sede_desc_ingles'])

# las columnas 
# 'nombre_titular, apellido_titular, cargo_titular, 
# telefono_principal, telefonos_adicionales, celular_de_guardia, 
# celulares_adicionales, fax_principal, faxes_adicionales, sitio_web, sitios_web_adicionales, comentario_del_horario,
# correo_electronico, correos_adicionales, atencion_dia_desde, atencion_dia_hasta, atencion_hora_desde, atencion_hora_hasta' 
# no son necesarias ya que no vamos a analizar los datos de contacto ni de los titulares de las sedes
df_secciones = df_secciones.drop(columns=[
    'nombre_titular', 'apellido_titular', 'cargo_titular', 'telefono_principal', 'telefonos_adicionales', 
    'celular_de_guardia', 'celulares_adicionales', 'fax_principal', 'faxes_adicionales', 
    'sitio_web', 'sitios_web_adicionales', 'comentario_del_horario',
    'correo_electronico', 'correos_adicionales', 'atencion_dia_desde', 'atencion_dia_hasta', 
    'atencion_hora_desde', 'atencion_hora_hasta'])

# también sacamos la columna temas, ya que está vacía
df_secciones = df_secciones.drop(columns=['temas'])

 
# valores unicos para concurrencias
#print("Valores de la columna 'concurrencias':", df_sedes_datos['concurrencias'].unique())

 
# LISTA-SEDES-DATOS

# las columnas 'sede_desc_ingles', 'ciudad_ingles' y 'pais_ingles' no es necesaria ya que ya tenemos el nombre en español
df_sedes_datos = df_sedes_datos.drop(columns=['sede_desc_ingles', 'ciudad_ingles', 'pais_ingles'])

# las columnas 'pais_iso_2' y 'pais_iso_3' y 'codigo_postal', 'telefono_principal' no son necesarias para ninguna consulta
df_sedes_datos = df_sedes_datos.drop(columns=['pais_iso_2', 'codigo_postal','telefono_principal'])

# la columna 'pais_codigo_telefonico' no es necesaria para ninguna consulta
df_sedes_datos = df_sedes_datos.drop(columns=['pais_codigo_telefonico'])

# las columnas 'ciudad_zona_horaria_gmt	ciudad_codigo_telefonico titular_nombre	titular_apellido	titular_cargo	direccion' no son necesarias para ninguna consulta
df_sedes_datos = df_sedes_datos.drop(columns=['ciudad_zona_horaria_gmt', 'ciudad_codigo_telefonico', 'titular_nombre',	'titular_apellido',	'titular_cargo',	'direccion'])

# las columnas 'sitio_web	sitios_web_adicionales telefonos_adicionales	celular_guardia	celulares_adicionales	fax_principal	faxes_adicionales	correo_electronico	correos_electronicos_adicionales' no son necesarias para ninguna consulta
df_sedes_datos = df_sedes_datos.drop(columns=['sitio_web', 'sitios_web_adicionales', 'telefonos_adicionales',	'celular_guardia',	'celulares_adicionales',	'fax_principal',	'faxes_adicionales'	, 'correo_electronico',	'correos_electronicos_adicionales'])

# las columnas 'atencion_dia_desde	atencion_dia_hasta	atencion_hora_desde	atencion_hora_hasta	atencion_comentario' no son necesarias para ninguna consulta
df_sedes_datos = df_sedes_datos.drop(columns=['atencion_dia_desde', 'atencion_dia_hasta', 'atencion_hora_desde', 'atencion_hora_hasta', 'atencion_comentario'])

# # IMPORTACIÓN DE DATOS SEGÚN ESQUEMAS

 
#PAISES

# valores unicos en sedes columna pais_castellano
#print("Valores unicos en sedes columna pais_castellano:", df_sedes_datos['pais_castellano'].unique())
#print("Cantidad de valores distintos en la columna 'pais_castellano':", df_sedes_datos['pais_castellano'].nunique())

 
# un df con solo los valores de la columna 'pais_castellano' que no son nulos, y no son 'Argentinos en el exterior', y únicos, con el nombre de la columna 'pais'
df_paises = df_sedes_datos[df_sedes_datos['pais_castellano'].notna() & df_sedes_datos['region_geografica'].notna() & 
                     (df_sedes_datos['pais_castellano'] != 'Argentinos  en  el  exterior')][['pais_iso_3', 'pais_castellano', 'region_geografica']].drop_duplicates()
df_paises.columns = ['ISO3', 'nombre', 'region_geografica']

 
# filtrar df_migraciones donde el país de origen es el código de refugiados = zzz
""" df_migraciones = df_migraciones[
    (df_migraciones['Country Origin Code'] != 'zzz')] """
# en lugar de '1960 [1960] 1970 [1970] 1980 [1980] 1990 [1990] 2000 [2000]' usar 'año' y 'cantidad'

# renombrar '1960 [1960] 1970 [1970] 1980 [1980] 1990 [1990] 2000 [2000]' a solo el año
df_migraciones['1960'] = df_migraciones['1960 [1960]']
df_migraciones['1970'] = df_migraciones['1970 [1970]']
df_migraciones['1980'] = df_migraciones['1980 [1980]']
df_migraciones['1990'] = df_migraciones['1990 [1990]']
df_migraciones['2000'] = df_migraciones['2000 [2000]']
# eliminar las columnas '1960 [1960] 1970 [1970] 1980 [1980] 1990 [1990] 2000 [2000]'
df_migraciones = df_migraciones.drop(columns=['1960 [1960]', '1970 [1970]', '1980 [1980]', '1990 [1990]', '2000 [2000]'])

# Transformar el dataframe para convertir las columnas de años en filas
df_migraciones = df_migraciones.melt(id_vars=['Country Origin Code', 'Country Dest Code'], 
                                     var_name='año', 
                                     value_name='cantidad')

"""Country Origin Name Country Dest Name año cantidad
0 Afghanistan Argentina 1960 6"""
# si tenemos el nombre del país de destino como Argentina, usar 'pais' con el otro país y establecer 'emigracion' como 0 y 'inmigracion' como 'cantidad'

""" for index, row in df_migraciones.iterrows():
    if row['Country Dest Code'] == 'ARG':
        df_migraciones.at[index, 'pais'] = row['Country Origin Code']
        df_migraciones.at[index, 'inmigracion'] = row['cantidad']
        df_migraciones.at[index, 'emigracion'] = 0
    else:
        df_migraciones.at[index, 'pais'] = row['Country Dest Code']
        df_migraciones.at[index, 'emigracion'] = row['cantidad']
        df_migraciones.at[index, 'inmigracion'] = 0 """

# renombrar columnas country origin code y country dest code a ISO3_origen y ISO3_destino
df_migraciones = df_migraciones.rename(columns={'Country Origin Code': 'ISO3_origen', 'Country Dest Code': 'ISO3_destino'})

 
# para la columna 'redes_sociales' dividir los valores en palabras y para cada palabra, crear una nueva fila con los mismos valores de las otras columnas
df_sedes_datos_redes = df_sedes_datos[df_sedes_datos['redes_sociales'].notna()]
# Dividir la columna 'redes_sociales' en palabras y crear una nueva fila para cada palabra
rows = []
for index, row in df_sedes_datos_redes.iterrows():
    words = row['redes_sociales'].split()
    # eliminar la fila con el valor original
    df_sedes_datos_redes.drop(index)

    for word in words:
        new_row = row.copy()
        new_row['redes_sociales'] = word
        rows.append(new_row)

df_redes_sociales = pd.DataFrame(rows)
# eliminar si redes sociales es //
df_redes_sociales = df_redes_sociales[
    (df_redes_sociales['redes_sociales'] != '//') &
    df_redes_sociales['redes_sociales'].str.contains('http') &
    df_redes_sociales['redes_sociales'].str.contains('www')
    ]

# Usar un diccionario para mapear palabras clave a plataformas
platform_map = {
    'facebook': 'Facebook',
    'twitter': 'Twitter',
    'instagram': 'Instagram',
    'linkedin': 'Linkedin',
    'youtube': 'Youtube'
}

# Asignar plataforma basada en la presencia de palabras clave
df_redes_sociales.reset_index(drop=True, inplace=True)
for index, row in df_redes_sociales.iterrows():
    for keyword, platform in platform_map.items():
        if keyword in row['redes_sociales']:
            df_redes_sociales.at[index, 'plataforma'] = platform
            break
    else:
        df_redes_sociales.at[index, 'plataforma'] = 'Otras'

# renombrar redes_sociales a url
df_redes_sociales = df_redes_sociales.rename(columns={'redes_sociales': 'url'})

# eliminar sede_desc_castellano, region_geografica, ciudad_castellano, estado, concurrencias, circunscripcion
df_redes_sociales = df_redes_sociales.drop(columns=['sede_desc_castellano', 'region_geografica', 'ciudad_castellano', 'estado', 'concurrencias', 'circunscripcion', 'pais_castellano'])

# eliminar pais_iso_3
df_redes_sociales = df_redes_sociales.drop(columns=['pais_iso_3'])

df_redes_sociales = df_redes_sociales[['url', 'plataforma', 'sede_id']]

 
# eliminar sede_desc_ingles, sede_desc_castellano, pais_castellano, ciudad_castellano, estado, sede_tipo
df_sedes = df_sedes.drop(columns=['sede_desc_ingles', 'sede_desc_castellano', 'pais_castellano', 'pais_ingles', 'ciudad_castellano', 'estado', 'sede_tipo'])

 

# contar el número de secciones para cada sede_id
df_secciones['cantidad_secciones'] = 1
df_secciones_count = df_secciones.groupby('sede_id').sum()
df_secciones_count = df_secciones_count.reset_index()

 
# encontrar las sedes que no están en secciones count
df_sedes_no_secciones = df_sedes[~df_sedes['sede_id'].isin(df_secciones_count['sede_id'])]

# agregar cantidad_secciones = 0 a las sedes que no están en secciones count
df_sedes_no_secciones['cantidad_secciones'] = 0
# reiniciar ambos índices
df_sedes_no_secciones = df_sedes_no_secciones.reset_index(drop=True)
df_secciones_count = df_secciones_count.reset_index(drop=True)
# combinar las sedes con secciones count y sin secciones count
df_secciones_count = pd.concat([df_sedes_no_secciones, df_secciones_count])

 
# renombrar columna pais_iso_3 a ISO3 
df_secciones_count = df_secciones_count.rename(columns={'pais_iso_3': 'ISO3'})

df_sedes = df_secciones_count

# DITIC parece ser una sede de TESTING, la eliminamos
df_sedes = df_sedes[df_sedes['sede_id'] != 'DITIC']

for index, row in df_sedes[df_sedes['ISO3'].isna()].iterrows():
    for index2, row2 in df_sedes_datos[df_sedes_datos['sede_id'] == row['sede_id']].iterrows():
        df_sedes.at[index, 'ISO3'] = row2['pais_iso_3']

 
#migraciones renombrar
df_migraciones.rename(columns={'año': 'anio'}, inplace=True)

# reemplazar los valores de la columna 'cantidad' que son .. por 0
df_migraciones['cantidad'] = df_migraciones['cantidad'].replace('..', 0)

# sedes que tienen null en iso3
df_sedes_null_iso3 = df_sedes[df_sedes['ISO3'].isna()]
df_sedes_null_iso3.shape

 
#exportar los dataframes a csv a /exports
df_sedes.to_csv('exports/sedes.csv', index=False)
df_migraciones.to_csv('exports/emigracion.csv', index=False)
df_paises.to_csv('exports/paises.csv', index=False)
df_redes_sociales.to_csv('exports/redes_sociales.csv', index=False)

 
# imprimir heads
print(df_sedes.head())
print(df_migraciones.head())
print(df_paises.head())
print(df_redes_sociales.head())
