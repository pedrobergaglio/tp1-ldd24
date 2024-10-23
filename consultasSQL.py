import pandas as pd
from inline_sql import sql, sql_val

paises = pd.read_csv('esquemas/paises.csv')
migracion = pd.read_csv('esquemas/migracion.csv')
sedes = pd.read_csv('esquemas/sedes.csv')
redes = pd.read_csv('esquemas/redes_sociales.csv')

#%% i)
sedes_secciones = sql^ """
                SELECT ISO3, COUNT(sede_id) AS sedes, AVG(cantidad_secciones) AS secciones_promedio
                FROM sedes
                GROUP BY ISO3;
"""

emigra = sql^ """
            SELECT ISO3_origen, SUM(CAST(cantidad AS INTEGER)) AS cantidad
            FROM migracion
            WHERE anio='2000'
            GROUP BY ISO3_origen;
            
"""
inmigra = sql^ """
            SELECT ISO3_destino, SUM(CAST(cantidad AS INTEGER)) AS cantidad
            FROM migracion
            WHERE anio='2000'
            GROUP BY ISO3_destino;
"""

flujo_neto = sql^ """
            SELECT e.ISO3_origen AS ISO3, i.cantidad - e.cantidad AS neto
            FROM emigra AS e
            INNER JOIN inmigra AS i
            ON e.ISO3_origen=i.ISO3_destino;
"""

sedes_flujo = sql^ """
            SELECT s.ISO3, s.sedes, s.secciones_promedio, fn.neto
            FROM sedes_secciones AS s
            INNER JOIN flujo_neto AS fn
            ON s.ISO3=fn.ISO3;
"""

paises_flujo = """
            SELECT p.nombre AS Pais, sf.sedes, sf.secciones_promedio AS 'secciones promedio', sf.neto AS 'flujo migratorio neto'
            FROM sedes_flujo AS sf
            INNER JOIN paises AS p
            ON sf.ISO3=p.ISO3
            ORDER BY sf.sedes DESC, p.nombre ASC;
"""

ejercicio_i = sql^ paises_flujo

print(ejercicio_i)

# Exportamos el data frame a .csv en la carpeta 'consultas' 
ejercicio_i.to_csv('consultas/consulta_i.csv', index=False)

#%% ii)
paises_sedes = sql^"""
                    SELECT DISTINCT s.ISO3, region_geografica
                    FROM sedes AS s
                    INNER JOIN paises AS p
                    ON s.ISO3=p.ISO3;
"""

cantidad = sql^"""
                SELECT COUNT(ISO3) AS 'cant', region_geografica
                FROM paises_sedes 
                GROUP BY region_geografica;
"""

flujo_emigracion = sql^ """
                    SELECT cantidad, region_geografica
                    FROM migracion
                    INNER JOIN paises
                    ON ISO3_destino = ISO3
                    WHERE ISO3_origen = 'ARG';
"""

promedio_flujo = sql^ """
                    SELECT AVG(CAST(cantidad AS INTEGER)) as promedio, region_geografica
                    FROM flujo_emigracion
                    GROUP BY region_geografica;
"""

pais_promedio = """
                SELECT c.region_geografica AS 'Region Geografica', c.cant AS 'Paises Con Sedes Argentinas', pf.promedio AS 'Promedio flujo con Argentina - Países con Sedes Argentinas'
                FROM cantidad AS c
                INNER JOIN promedio_flujo AS pf
                ON c.region_geografica=pf.region_geografica
                ORDER BY pf.promedio DESC;
"""

ejercicio_ii = sql^ pais_promedio

print(ejercicio_ii)

# Exportamos el data frame a .csv en la carpeta 'consultas' 
ejercicio_ii.to_csv('consultas/consulta_ii.csv', index=False)

#%% iii)
sedes_redes = sql^"""
            SELECT DISTINCT plataforma, ISO3
            FROM sedes AS s
            INNER JOIN redes AS r
            ON s.sede_id=r.sede_id;
"""

redes_pais = sql^"""
            SELECT COUNT(plataforma) AS cantidad, ISO3
            FROM sedes_redes AS sr
            GROUP BY ISO3;
"""

cantidad_redes = """
                SELECT nombre AS Pais, cantidad AS 'Cantidad Redes'
                FROM redes_pais AS rp
                INNER JOIN paises AS p
                ON rp.ISO3=p.ISO3;
"""

ejercicio_iii = sql^ cantidad_redes

print(ejercicio_iii)

# Exportamos el data frame a .csv en la carpeta 'consultas' 
ejercicio_iii.to_csv('consultas/consulta_iii.csv', index=False)

#%% iv)
redes_sedes = sql^ """
               SELECT url, plataforma, r.sede_id, ISO3
               FROM redes AS r
               INNER JOIN sedes AS s
               ON r.sede_id=s.sede_id;
              """

redes_sociales = """
                SELECT nombre AS Pais, sede_id AS Sede, plataforma AS 'Red Social', url AS URL
                FROM redes_sedes AS r
                INNER JOIN paises AS p
                ON r.ISO3=p.ISO3
                ORDER BY nombre ASC, sede_id  ASC, plataforma ASC, url ASC;
"""

ejercicio_iv = sql^ redes_sociales

print(ejercicio_iv)

# Exportamos el data frame a .csv en la carpeta 'consultas' 
ejercicio_iv.to_csv('consultas/consulta_iv.csv', index=False)
