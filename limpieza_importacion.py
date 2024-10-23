# Importamos librerias y leemos los archivos csv
import pandas as pd

df_sedes = pd.read_csv('fuentes de datos/lista-sedes.csv')
df_migraciones = pd.read_csv('fuentes de datos/datos_migraciones.csv')
df_secciones = pd.read_csv('fuentes de datos/lista-secciones.csv')
# hay una fila que falla, por lo que se ignoran los errores
df_sedes_datos = pd.read_csv('fuentes de datos/lista-sedes-datos.csv', sep=',', on_bad_lines='skip')

#%% Calculos realizados para la metrica GQM

# Cantidad de veces que aparece el valor 'Argentinos  en  el  exterior' en la columna 'ciudad' de la tabla df_lista_sedes
df_sedes_ciudad = df_sedes[df_sedes['ciudad_castellano'] == 'Argentinos  en  el  exterior']
print('Cantidad de veces que aparece el valor "Argentinos  en  el  exterior" en la columna "ciudad_castellano" de la tabla df_lista_sedes: ', df_sedes_ciudad.shape[0])
# Cantidad de sedes
print("Cantidad de sedes: ", df_sedes.shape[0])

# Cantidad de valores en migraciones que tienen .. en la columna '2000 [2000]'
df_migraciones_2000 = df_migraciones[df_migraciones['2000 [2000]'] == '..']
print('Cantidad de valores en migraciones que tienen ".." en la columna "2000 [2000]": ', df_migraciones_2000.shape[0])
# Cantidad total
print('Cantidad total de valores en migraciones: ', df_migraciones.shape[0])
 
# Cantidad de valores que tienen caracteres no dígitos o - en la columna 'codigo_postal' en la tabla lista-sedes-datos.csv
df_sedes_datos_codigo_postal = df_sedes_datos[df_sedes_datos['codigo_postal'].str.contains(r'\D|-', na=False)]
print('Cantidad de valores que tienen caracteres no dígitos o - en la columna "codigo_postal" en la tabla lista-sedes-datos.csv: ', df_sedes_datos_codigo_postal.shape[0])
# Cantidad de sedes
print("Cantidad de sedes: ", df_sedes.shape[0])
 
# Cantidad de valores que no son nulos, y no tienen una dirección de correo válida (una sola palabra y tiene @) en la columna 'correo_electronico' en la tabla lista-secciones.csv
df_secciones_no_validas = df_secciones[
    df_secciones['correo_electronico'].notna() &
    (df_secciones['correo_electronico'].astype(str).str.contains(" ") |
    ~df_secciones['correo_electronico'].astype(str).str.contains("@"))
    ]
print('Cantidad de valores que tienen direcciones no válidas en la columna "correo_electronico" en la tabla lista-secciones.csv: ', df_secciones_no_validas.shape[0])
# Cantidad de secciones
print("Cantidad de secciones: ", df_secciones.shape[0])

#%% Eliminamos todas aquellas columnas que sabemos que no vamos a utilizar

# MIGRACIONES
# Las columnas con nombres de paises son redundantes al igual que el codigo de genero, las podemos eliminar
df_migraciones = df_migraciones.drop(columns=['Migration by Gender Code', 'Country Origin Name', 'Country Dest Name'])

# La columna 'Migration by Gender Code' no es necesaria para las consultas, basta con utilizar las filas que tienen Total # filtramos las filas que tienen Total en 'Migration by Gender Name'
df_migraciones = df_migraciones[df_migraciones['Migration by Gender Name'] == 'Total']
# Eliminamos la columna 'Migration by Gender Name'
df_migraciones = df_migraciones.drop(columns=['Migration by Gender Name'])
 
# Chequeamos valores unicos en columnas sede_tipo y estado
print("Valores unicos en sedes columna sede_tipo:", df_sedes['sede_tipo'].unique())
print("Valores unicos en sedes columna estado:", df_sedes['estado'].unique())


# SEDES
# Las columnas sede_desc_ingles, pais_iso_2, pais_iso_3, pais_ingles, ciudad_ingles, no son necesarias para las consultas
df_sedes = df_sedes.drop(columns=['pais_iso_2', 'ciudad_ingles'])
 
# Chequeamos valores unicos en la columna temas 
print("Valores de la columna 'temas':", df_secciones['temas'].unique())


# SECCIONES
# Las columnas 'sede_desc_ingles' no es necesaria ya que tenemos el nombre en español
df_secciones = df_secciones.drop(columns=['sede_desc_ingles'])

# Las columnas 'nombre_titular, apellido_titular, cargo_titular, telefono_principal, telefonos_adicionales, celular_de_guardia, celulares_adicionales, fax_principal, faxes_adicionales, sitio_web, sitios_web_adicionales, comentario_del_horario, correo_electronico, correos_adicionales, atencion_dia_desde, atencion_dia_hasta, atencion_hora_desde, atencion_hora_hasta' no son necesarias ya que no vamos a analizar los datos de contacto ni de los titulares de las sedes
df_secciones = df_secciones.drop(columns=[
    'nombre_titular', 'apellido_titular', 'cargo_titular', 'telefono_principal', 'telefonos_adicionales', 
    'celular_de_guardia', 'celulares_adicionales', 'fax_principal', 'faxes_adicionales', 
    'sitio_web', 'sitios_web_adicionales', 'comentario_del_horario',
    'correo_electronico', 'correos_adicionales', 'atencion_dia_desde', 'atencion_dia_hasta', 
    'atencion_hora_desde', 'atencion_hora_hasta'])

# También sacamos la columna temas, ya que está vacía
df_secciones = df_secciones.drop(columns=['temas'])

# Chequeamos valores unicos para para la columna concurrencias
print("Valores de la columna 'concurrencias':", df_sedes_datos['concurrencias'].unique())

 
# SEDES COMPLETA
# Las columnas 'sede_desc_ingles', 'ciudad_ingles' y 'pais_ingles' no es necesaria ya que ya tenemos el nombre en español
df_sedes_datos = df_sedes_datos.drop(columns=['sede_desc_ingles', 'ciudad_ingles', 'pais_ingles'])

# Las columnas 'pais_iso_2' y 'pais_iso_3' y 'codigo_postal', 'telefono_principal' no son necesarias para ninguna consulta
df_sedes_datos = df_sedes_datos.drop(columns=['pais_iso_2', 'codigo_postal','telefono_principal'])

# La columna 'pais_codigo_telefonico' no es necesaria para ninguna consulta
df_sedes_datos = df_sedes_datos.drop(columns=['pais_codigo_telefonico'])

# Las columnas 'ciudad_zona_horaria_gmt', 'ciudad_codigo_telefonico', 'titular_nombre',	'titular_apellido',	'titular_cargo', 'direccion' no son necesarias para ninguna consulta
df_sedes_datos = df_sedes_datos.drop(columns=['ciudad_zona_horaria_gmt', 'ciudad_codigo_telefonico', 'titular_nombre',	'titular_apellido',	'titular_cargo',	'direccion'])

# Las columnas 'sitio_web', 'sitios_web_adicionales', 'telefonos_adicionales	', 'celular_guardia', 'celulares_adicionales', 'fax_principal', 'faxes_adicionales', 'correo_electronico', 'correos_electronicos_adicionales' no son necesarias para ninguna consulta
df_sedes_datos = df_sedes_datos.drop(columns=['sitio_web', 'sitios_web_adicionales', 'telefonos_adicionales',	'celular_guardia',	'celulares_adicionales',	'fax_principal',	'faxes_adicionales'	, 'correo_electronico',	'correos_electronicos_adicionales'])

# Las columnas 'atencion_dia_desde', 'atencion_dia_hasta', 'atencion_hora_desde', 'atencion_hora_hasta', 'atencion_comentario' no son necesarias para ninguna consulta
df_sedes_datos = df_sedes_datos.drop(columns=['atencion_dia_desde', 'atencion_dia_hasta', 'atencion_hora_desde', 'atencion_hora_hasta', 'atencion_comentario'])


#%% Realizamos la importacion de datos segun los esquemas de nuestro Modelo Relacional
 
# PAISES
# Chequeamos valores en la columna pais_castellano
print("Valores unicos en sedes columna pais_castellano:", df_sedes_datos['pais_castellano'].unique())
print("Cantidad de valores distintos en la columna 'pais_castellano':", df_sedes_datos['pais_castellano'].nunique())

# Creamos un df a partir de df_sedes_datos que contenga unicamente los valores de la columna 'pais_castellano' que no son nulos, y no son 'Argentinos en el exterior', y únicos, con el nombre de la columna 'pais'
df_paises = df_sedes_datos[df_sedes_datos['pais_castellano'].notna() & df_sedes_datos['region_geografica'].notna() & 
                     (df_sedes_datos['pais_castellano'] != 'Argentinos  en  el  exterior')][['pais_iso_3', 'pais_castellano', 'region_geografica']].drop_duplicates()
# Renombramos las columnas
df_paises.columns = ['ISO3', 'nombre', 'region_geografica']

 
# MIGRACIONES
# A partir de df_migraciones en lugar de '1960 [1960] 1970 [1970] 1980 [1980] 1990 [1990] 2000 [2000]' usamos 'año' y 'cantidad'
# Renombramos '1960 [1960]', '1970 [1970]', '1980 [1980]', '1990 [1990]', '2000 [2000]' a solo el año
df_migraciones['1960'] = df_migraciones['1960 [1960]']
df_migraciones['1970'] = df_migraciones['1970 [1970]']
df_migraciones['1980'] = df_migraciones['1980 [1980]']
df_migraciones['1990'] = df_migraciones['1990 [1990]']
df_migraciones['2000'] = df_migraciones['2000 [2000]']

# Eliminamos las columnas '1960 [1960]', '1970 [1970]', '1980 [1980]', '1990 [1990]', '2000 [2000]'
df_migraciones = df_migraciones.drop(columns=['1960 [1960]', '1970 [1970]', '1980 [1980]', '1990 [1990]', '2000 [2000]'])

# Transformamos el dataframe para convertir las columnas de años en filas
df_migraciones = df_migraciones.melt(id_vars=['Country Origin Code', 'Country Dest Code'], 
                                     var_name='anio', 
                                     value_name='cantidad')

# Renombramos las columnas country origin code y country dest code a ISO3_origen e ISO3_destino
df_migraciones = df_migraciones.rename(columns={'Country Origin Code': 'ISO3_origen', 'Country Dest Code': 'ISO3_destino'})

# Reemplazamos los valores de la columna 'cantidad' que son .. por 0
df_migraciones['cantidad'] = df_migraciones['cantidad'].replace('..', 0)


# REDES SOCIALES
# Utilizamos el data frame df_sedes_datos para obtener las redes sociales de las distintas secciones, nos quedamos unicamente con aquellas filas que no tengan redes sociales nulas
df_sedes_datos_redes = df_sedes_datos[df_sedes_datos['redes_sociales'].notna()]

# Dividimos la columna 'redes_sociales' en palabras y creamos una nueva fila para cada palabra
rows = []
for index, row in df_sedes_datos_redes.iterrows():
    words = row['redes_sociales'].split()
    # Eliminamos la fila con el valor original
    df_sedes_datos_redes.drop(index)

    for word in words:
        new_row = row.copy()
        new_row['redes_sociales'] = word
        rows.append(new_row)

df_redes_sociales = pd.DataFrame(rows)
# Eliminamos aquellas redes sociales que sean solo '//'
df_redes_sociales = df_redes_sociales[
    (df_redes_sociales['redes_sociales'] != '//') &
    df_redes_sociales['redes_sociales'].str.contains('http') &
    df_redes_sociales['redes_sociales'].str.contains('www')
    ]

# Usamos un diccionario para mapear palabras clave a plataformas
platform_map = {
    'facebook': 'Facebook',
    'twitter': 'Twitter',
    'instagram': 'Instagram',
    'linkedin': 'Linkedin',
    'youtube': 'Youtube'
}

# Asignamos la plataforma basandonos en la presencia de las palabras clave
df_redes_sociales.reset_index(drop=True, inplace=True)
for index, row in df_redes_sociales.iterrows():
    for keyword, platform in platform_map.items():
        if keyword in row['redes_sociales']:
            df_redes_sociales.at[index, 'plataforma'] = platform
            break
    else:
        df_redes_sociales.at[index, 'plataforma'] = 'Otras'

# Renombramos redes_sociales a url
df_redes_sociales = df_redes_sociales.rename(columns={'redes_sociales': 'url'})

# Eliminamos 'sede_desc_castellano', 'region_geografica', 'ciudad_castellano', 'estado', 'concurrencias', 'circunscripcion'
df_redes_sociales = df_redes_sociales.drop(columns=['sede_desc_castellano', 'region_geografica', 'ciudad_castellano', 'estado', 'concurrencias', 'circunscripcion', 'pais_castellano'])

# Eliminamos pais_iso_3
df_redes_sociales = df_redes_sociales.drop(columns=['pais_iso_3'])

# Renombramos las columnas
df_redes_sociales = df_redes_sociales[['url', 'plataforma', 'sede_id']]


# SEDES
# Eliminamos 'sede_desc_ingles', 'sede_desc_castellano', 'pais_castellano', 'ciudad_castellano', 'estado', 'sede_tipo'
df_sedes = df_sedes.drop(columns=['sede_desc_ingles', 'sede_desc_castellano', 'pais_castellano', 'pais_ingles', 'ciudad_castellano', 'estado', 'sede_tipo'])

# Contamos el número de secciones para cada sede_id
df_secciones['cantidad_secciones'] = 1
df_secciones_count = df_secciones.groupby('sede_id').sum()
df_secciones_count = df_secciones_count.reset_index()

# Encontramos las sedes que no están en df_secciones_count (aquellas que no tienen secciones)
df_sedes_no_secciones = df_sedes[~df_sedes['sede_id'].isin(df_secciones_count['sede_id'])]

# Agregamos cantidad_secciones = 0 a las sedes que no están en df_secciones_count
df_sedes_no_secciones['cantidad_secciones'] = 0

# Reiniciamos ambos índices
df_sedes_no_secciones = df_sedes_no_secciones.reset_index(drop=True)
df_secciones_count = df_secciones_count.reset_index(drop=True)

# Combinamos las sedes que estan en secciones count y sin secciones count
df_secciones_count = pd.concat([df_sedes_no_secciones, df_secciones_count])

# Renombramos la columna pais_iso_3 a ISO3 
df_secciones_count = df_secciones_count.rename(columns={'pais_iso_3': 'ISO3'})

# Eliminamos sede_desc_castellano y tipo_seccion
df_secciones_count = df_secciones_count.drop(columns=['sede_desc_castellano', 'tipo_seccion'])

# Renombramos el dataframe
df_sedes = df_secciones_count

# DITIC parece ser una sede de TESTING, la eliminamos
df_sedes = df_sedes[df_sedes['sede_id'] != 'DITIC']

for index, row in df_sedes[df_sedes['ISO3'].isna()].iterrows():
    for index2, row2 in df_sedes_datos[df_sedes_datos['sede_id'] == row['sede_id']].iterrows():
        df_sedes.at[index, 'ISO3'] = row2['pais_iso_3']

# Sedes que tienen null en iso3
df_sedes_null_iso3 = df_sedes[df_sedes['ISO3'].isna()]
df_sedes_null_iso3.shape


#%% Exportamos los dataframes a .csv en la carpeta 'esquemas' 
df_sedes.to_csv('esquemas/sedes.csv', index=False)
df_migraciones.to_csv('esquemas/migracion.csv', index=False)
df_paises.to_csv('esquemas/paises.csv', index=False)
df_redes_sociales.to_csv('esquemas/redes_sociales.csv', index=False)
