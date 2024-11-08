#%%
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import statsmodels.formula.api  as smf
import statsmodels.api as sm
import numpy as np

#%%
# Importa el archivo como dataframe
carpeta = ""
df = pd.read_csv(carpeta+"Alturas.csv", sep=",")

#%%
df.head()

#%%
df.rename(columns={df.columns[3]: 'contextura_mama', df.columns[4]: 'altura_mama'}, inplace=True)

#%%
# Armamos un dataframe para cada variable
df = pd.get_dummies(df, columns=['sexo'], drop_first=True)

#%%
df.head()

#%%
df['sexo_M'] = df['sexo_M'].astype(int)

#%%
# Definir la variable dependiente e independientes
x = df[['altura_mama', 'sexo_M']]  # 'sexo_F' representa el sexo femenino
y = df['altura']

#%%
# Agregar una constante para la intersección en el modelo
x = sm.add_constant(x)

# Ajustar el modelo usando statsmodels
model = sm.OLS(y, x).fit()
summary = model.summary()

print(summary)


#%%
residuos = model.resid

# Mostrar los residuos
print("Residuos:", residuos)

#%%
residuos[0]

#%%
df['prediccion'] = model.predict(x)
df['altura'][0] - df['prediccion'][0]

#%%
# Crear QQ-plot
sm.qqplot(residuos, line='s')
plt.title('QQ-Plot de Residuos')
plt.show()

#%%
# Predecir un nuevo dato
altura_nueva = 165  # Por ejemplo, 165 cm
sexo_M = 1         # 1 para masculino, 0 para femenino

#%%
# Crear una lista con los valores de las variables para la predicción incluyendo la constante, altura_nueva y el nivel de sexo

#para saber el orden de ingreso de las variables
print(model.model.exog_names)

nuevas_observaciones = [1, altura_nueva, sexo_M]  # [constante, altura_nueva, sexo_M]

#%%
# Hacer la predicción
prediccion = model.predict(nuevas_observaciones)

#%%
# Mostrar la predicción
print(prediccion)


#%%
#Vamos a hacer un modelo con interaccion
df['interaccion'] = df['altura_mama'] * df['sexo_M']

#%%
x = df[['altura_mama', 'sexo_M', 'interaccion']]
y = df['altura']

#%%
# Agregar una constante al modelo (término independiente)
x = sm.add_constant(x)
# Ajustar el nuevo modelo OLS
model = sm.OLS(y, x).fit()

#%%
# Mostrar el resumen del modelo
summary = model.summary()
print(summary)

#%%
##################################################################
##################################################################
#otra forma mucho mas facil
# import statsmodels.formula.api as smf
modelo = smf.ols('altura ~ altura_mama * sexo', data = df).fit()  # 
print(modelo.summary())
###################################################################
###################################################################

#%%
# Extraer los coeficientes
coef = model.params
intercepto = coef['const']
beta_altura_mama = coef['altura_mama']
beta_sexo_M = coef['sexo_M']
beta_interaccion = coef['interaccion']

# Calcular las predicciones para los datos originales
df['prediccion'] = model.predict(x)

# Graficar los datos originales en función del sexo
plt.figure(figsize=(10, 6))

# Separar puntos según el sexo
fem_data = df[df['sexo_M'] == 0]
male_data = df[df['sexo_M'] == 1]

# Graficar puntos originales por sexo
plt.scatter(fem_data['altura_mama'], fem_data['altura'], color='purple', label='Femenino')
plt.scatter(male_data['altura_mama'], male_data['altura'], color='blue', label='Masculino')

# Definir un rango de valores de altura_mama para calcular las rectas
altura_mama_range = np.linspace(df['altura_mama'].min(), df['altura_mama'].max(), 100)

# Calcular y graficar la recta para el sexo femenino (sexo_M = 0)
predicciones_fem = (intercepto + 
                    beta_altura_mama * altura_mama_range + 
                    beta_sexo_M * 0 + 
                    beta_interaccion * (altura_mama_range * 0))
plt.plot(altura_mama_range, predicciones_fem, color='purple', linestyle='--', label='Recta Estimada (Femenino)')

# Calcular y graficar la recta para el sexo masculino (sexo_M = 1)
predicciones_male = (intercepto + 
                     beta_altura_mama * altura_mama_range + 
                     beta_sexo_M * 1 + 
                     beta_interaccion * (altura_mama_range * 1))
plt.plot(altura_mama_range, predicciones_male, color='blue', linestyle='-', label='Recta Estimada (Masculino)')

# Etiquetas y título
plt.xlabel('Altura de la madre (cm)')
plt.ylabel('Altura del hijo (cm)')
plt.title('Predicción de altura en función de la altura de la madre y el sexo')
plt.legend()
plt.grid()

# Mostrar el gráfico
plt.show()