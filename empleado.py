from inline_sql import sql, sql_val
import pandas as pd

# Ejercicios AR-PROJECT, SELECT, RENAME
empleado = pd.read_csv("empleado.csv")

print(empleado)

consultaSQL = """
               SELECT DISTINCT DNI, Salario
               FROM empleado;
              """

dataframeResultado = sql^ consultaSQL

print(dataframeResultado)