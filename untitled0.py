#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:42:34 2024

@author: Estudiante
"""

import pandas as pd
from inline_sql import sql, sql_val

pais = pd.read_csv('/home/Estudiante/Descargas/archivos/paises.csv')
emigracion = pd.read_csv('/home/Estudiante/Descargas/archivos/emigracion.csv')
sedes = pd.read_csv('/home/Estudiante/Descargas/archivos/sedes.csv')
redes = pd.read_csv('/home/Estudiante/Descargas/archivos/redes_sociales.csv')

#%%
paises_sedes = sql^"""
                    SELECT DISTINCT s.ISO3, region_geografica
                    FROM sedes AS s
                    INNER JOIN pais AS p
                    ON s.ISO3=p.ISO3;
"""

cantidad = sql^"""
                SELECT COUNT(ISO3) AS 'Paises Con Sedes Argentinas', region_geografica
                FROM paises_sedes AS ps
                GROUP BY region_geografica;
"""

#%% 
sedes_redes = sql^"""
            SELECT plataforma, ISO3
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
                INNER JOIN pais AS p
                ON rp.ISO3=p.ISO3;
"""

ejercicio_iii = sql^ cantidad_redes

print(ejercicio_iii)

#%%
redes_sedes = sql^ """
               SELECT url, plataforma, r.sede_id, ISO3
               FROM redes AS r
               INNER JOIN sedes AS s
               ON r.sede_id=s.sede_id;
              """

redes_sociales = """
                SELECT nombre, sede_id, plataforma, url
                FROM redes_sedes AS r
                INNER JOIN pais AS p
                ON r.ISO3=p.ISO3
                ORDER BY nombre ASC, sede_id  ASC, plataforma ASC, url ASC;
"""

ejercicio_iv = sql^ redes_sociales

print(ejercicio_iv)