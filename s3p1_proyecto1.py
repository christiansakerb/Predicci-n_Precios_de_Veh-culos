# -*- coding: utf-8 -*-
"""S3P1_proyecto1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DAAJh1rG9LlbquftJQ71F1XiFfSAv4Zz

![image info](https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/images/banner_1.png)

# Proyecto 1 - Predicción de precios de vehículos usados

En este proyecto podrán poner en práctica sus conocimientos sobre modelos predictivos basados en árboles y ensambles, y sobre la disponibilización de modelos. Para su desasrrollo tengan en cuenta las instrucciones dadas en la "Guía del proyecto 1: Predicción de precios de vehículos usados".

**Entrega**: La entrega del proyecto deberán realizarla durante la semana 4. Sin embargo, es importante que avancen en la semana 3 en el modelado del problema y en parte del informe, tal y como se les indicó en la guía.

Para hacer la entrega, deberán adjuntar el informe autocontenido en PDF a la actividad de entrega del proyecto que encontrarán en la semana 4, y subir el archivo de predicciones a la [competencia de Kaggle](https://www.kaggle.com/t/b8be43cf89c540bfaf3831f2c8506614).

## Datos para la predicción de precios de vehículos usados

En este proyecto se usará el conjunto de datos de Car Listings de Kaggle, donde cada observación representa el precio de un automóvil teniendo en cuenta distintas variables como: año, marca, modelo, entre otras. El objetivo es predecir el precio del automóvil. Para más detalles puede visitar el siguiente enlace: [datos](https://www.kaggle.com/jpayne/852k-used-car-listings).

## Ejemplo predicción conjunto de test para envío a Kaggle

En esta sección encontrarán el formato en el que deben guardar los resultados de la predicción para que puedan subirlos a la competencia en Kaggle.
"""

import warnings
warnings.filterwarnings('ignore')

# Importación librerías
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Carga de datos de archivo .csv
dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)

plt.plot(np.cumsum(dataTraining.value_counts('Make',normalize=True)).to_numpy())

Umbral = np.sum(np.cumsum(dataTraining.value_counts('Make',normalize=True)).to_numpy() < 0.8)
plt.axhline(y=0.9, color='r')
plt.axvline(x=Umbral)
print(Umbral)
plt.title('Pareto de frecuencia de Makeos de vehículos')
plt.show()

Info_Make_models= dataTraining.apply(lambda x: x['Make']+x['Model'], axis=1).drop_duplicates().tolist()
Frecuencia_Acumulada = dataTraining[['Make','Model']].value_counts(normalize=True).reset_index().sort_values(0,ascending=False)
Frecuencia_Acumulada['Porcentaje Acum'] = Frecuencia_Acumulada[0].cumsum()
Particiones = [0.2,0.4,0.6,0.8,0.9,0.95,0.985,1]
Particiones_Marcas = {}
for i in Particiones:
    Particiones_Marcas[i]= Frecuencia_Acumulada[Frecuencia_Acumulada['Porcentaje Acum']<i][['Make','Model']]
    Frecuencia_Acumulada = Frecuencia_Acumulada[Frecuencia_Acumulada['Porcentaje Acum']>=i]

df_models = {}
suma=0
for i in Particiones:
    df_aux=pd.merge(Particiones_Marcas[i],dataTraining,on=['Make','Model'])
    df_aux['Make-Mod'] = df_aux.apply(lambda x: x['Make']+x['Model'], axis=1)
    df_aux=df_aux.drop(columns=['Make','Model'],axis=1)
    df_models[i]=df_aux
    print('La cantidad de registros de la partición ' + str(i) + ' es ' + str(df_aux.shape))
    suma += df_aux.shape[0]
    print('Cantidad total de registros: ' +str (suma))
df_models[0.2].head(5)

#Escalado de variables
df_scaled = {}
scalers ={}
valores_maximos= {}
valores_minimos = {}
Variable_y = ['Price']
Variables_numericas=['Year','Mileage']
Variables_categoricas= ['State','Make-Mod']
for i in Particiones:
    df_model_nums = df_models[i][Variables_numericas]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_model_nums)
    scalers[i]= scaler
    scaled_data = pd.DataFrame(scaled_data,columns=['Year','Mileage'])
    valores_maximos[i]=scaled_data.max().tolist()
    valores_minimos[i]=scaled_data.min().tolist()

    df_scaled[i]=pd.concat([df_models[i][Variable_y+Variables_categoricas],scaled_data],axis=1)
df_scaled[0.2].head()

#Codificación One-Hot
df_dummies = {}
for i in Particiones:
    df_dummies[i]=pd.get_dummies(df_scaled[i],columns=Variables_categoricas)
    print('La partición ' + str(i) + ' tiene la siguiente forma: ' + str(df_dummies[i].shape))
df_dummies[0.2]

#Separación Variables predictoras y a predecir.
df_transf = {}
for i in Particiones:
    TipoVariable = {}
    TipoVariable['X']=df_dummies[i].drop('Price',axis=1)
    TipoVariable['y']=df_dummies[i]['Price']
    df_transf[i]=TipoVariable
df_transf[0.2]['y']

"""#### Entrenamiento de modelos"""

#Cálculo de métricas
def Metricas(yTesteo,y_pred):
    mse_metric = round(mean_squared_error(yTesteo, y_pred),2)
    r2_metric = round(r2_score(yTesteo, y_pred),3)
    mae_metric = round(mean_absolute_error(yTesteo, y_pred),2)
    print("Desempeño del modelo:")
    print("Mean Squared Error:", mse_metric)
    print("R^2 Score:", r2_metric)
    print("Mean Absolute Error (MAE):", mae_metric)
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mape_metric = round(mean_absolute_percentage_error(yTesteo, y_pred),2)
    print("Mean Absolute Percentage Error (MAPE):", mape_metric, "%")
    print(' ')
    return [r2_metric,mse_metric,mae_metric,mape_metric]

"""##### XGBoost sin calibrar"""

#Entrenamiento de modelos
Xgboost_predictores = {}
Xgboost_metricas = {}
s=0
yTest_Completo = np.array([])
yPred_Completo = np.array([])
for i in Particiones:
    s+=1
    XTrain, XTest, yTrain, yTest = train_test_split(df_transf[i]['X'], df_transf[i]['y'], test_size=0.3, random_state=0)
    Xgboost_predictores[i] = XGBRegressor(random_state=0,n_estimators = 100,learning_rate=0.3,max_depth=6).fit(XTrain,yTrain)
    print('Modelo de partición ' + str(i) + ' Progreso: ' + str(round(s/len(Particiones)*100,0)) + '%')
    print('Dim Xtrain: ' + str(XTrain.shape)+ 'Dim Xtrain: ' + str(yTrain.shape)+ 'Dim Xtrain: ' + str(XTest.shape)+ 'Dim Xtrain: ' + str(yTest.shape))
    
    yPred = Xgboost_predictores[i].predict(XTest)
    yTest_Completo = np.concatenate((yTest_Completo,yTest.to_numpy()))
    yPred_Completo = np.concatenate((yPred_Completo,yPred))
    print('Resultados: ')
    Xgboost_metricas[i]=Metricas(yTest,yPred)
    #Desempeño modelo: 
print('Resultados finales Concatenados:')
Xgboost_metricas_Total = Metricas(yTest_Completo,yPred_Completo)

#Evuación de particiones
pd.DataFrame(Xgboost_metricas).T.rename(columns={0:'R2',1:'MSE',2:'MAE',3:'MAPE'})

"""##### Random Forest"""

#Entrenamiento de modelos
rf_predictores = {}
rf_metricas = {}
s=0
yTest_Completo = np.array([])
yPred_Completo = np.array([])
for i in Particiones:
    s+=1
    XTrain, XTest, yTrain, yTest = train_test_split(df_transf[i]['X'], df_transf[i]['y'], test_size=0.3, random_state=0)
    rf_predictores[i] = RandomForestRegressor(random_state=0).fit(XTrain,yTrain)
    print('Modelo de partición ' + str(i) + ' Progreso: ' + str(round(s/len(Particiones)*100,0)) + '%')
    print('Dim Xtrain: ' + str(XTrain.shape)+ 'Dim Xtrain: ' + str(yTrain.shape)+ 'Dim Xtrain: ' + str(XTest.shape)+ 'Dim Xtrain: ' + str(yTest.shape))
    
    yPred = rf_predictores[i].predict(XTest)
    yTest_Completo = np.concatenate((yTest_Completo,yTest.to_numpy()))
    yPred_Completo = np.concatenate((yPred_Completo,yPred))
    print('Resultados: ')
    rf_metricas[i]=Metricas(yTest,yPred)
    #Desempeño modelo: 
print('Resultados finales Concatenados:')
Random_Forest_metricas_Total= Metricas(yTest_Completo,yPred_Completo)

#Evuación de particiones
pd.DataFrame(rf_metricas).T.rename(columns={0:'R2',1:'MSE',2:'MAE',3:'MAPE'})

"""#### Calibración de modelo"""

#Calibrando parametros

def Probar_modelo(modelo):
    Xgboost_calibrado_predictores = {}
    Xgboost_calibrado_metricas = {}
    s=0
    yTest_Completo = np.array([])
    yPred_Completo = np.array([])
    for i in Particiones:
        s+=1
        XTrain, XTest, yTrain, yTest = train_test_split(df_transf[i]['X'], df_transf[i]['y'], test_size=0.3, random_state=0)
        Xgboost_calibrado_predictores[i] = modelo.fit(XTrain,yTrain)
        yPred = Xgboost_calibrado_predictores[i].predict(XTest)
        yTest_Completo = np.concatenate((yTest_Completo,yTest.to_numpy()))
        yPred_Completo = np.concatenate((yPred_Completo,yPred)) 
    return round(r2_score(yTest_Completo, yPred_Completo),3)
Probar_modelo(XGBRegressor(random_state=0))

#Calibración de Learning Rate:
learningRate_C = np.linspace(0.2, 0.4, num=5)
r2_learningRate_C = []

for k in learningRate_C:
    r2_learningRate_C.append(Probar_modelo(XGBRegressor(random_state=0,learning_rate=k)))

ymax = max(r2_learningRate_C)
xmax = learningRate_C.tolist()[r2_learningRate_C.index(ymax)]

plt.plot(learningRate_C,r2_learningRate_C)
plt.text(x=xmax,y=ymax, s=str(round(xmax,3))+' r2:' + str(ymax))

plt.title('Tasa de Aprendizaje')
plt.show()

#Calibración de N_Estimator:
n_estimators_C = np.linspace(100, 220, num=4, dtype=int)
r2_n_estimators_C = []
for k in n_estimators_C:
    r2_n_estimators_C.append(Probar_modelo(XGBRegressor(random_state=0,n_estimators=k)))

ymax = max(r2_n_estimators_C)
xmax = n_estimators_C.tolist()[r2_n_estimators_C.index(ymax)]

plt.plot(n_estimators_C,r2_n_estimators_C)
plt.text(x=xmax,y=ymax, s=str(xmax)+' r2:' + str(ymax))
plt.title('Número de estimadores')
plt.show()

#PRofundidad Máxima de arbol
max_depth_C = np.linspace(3, 12, num=4, dtype=int)
r2_max_depth_C = []
for k in max_depth_C:
    r2_max_depth_C.append(Probar_modelo(XGBRegressor(random_state=0,max_depth =k)))

ymax = max(r2_max_depth_C)
xmax = max_depth_C.tolist()[r2_max_depth_C.index(ymax)]

plt.plot(max_depth_C,r2_max_depth_C)
plt.text(x=xmax,y=ymax, s=str(xmax)+' r2:' + str(ymax))
plt.title('Profundidad de arboles')
plt.show()

"""#### Entrenamineto de modelo Calibrado"""

#Entrenamiento de modelos
Xgboost_calibrado_predictores = {}
Xgboost_calibrado_metricas = {}
s=0
yTest_Completo = np.array([])
yPred_Completo = np.array([])
for i in Particiones:
    s+=1
    XTrain, XTest, yTrain, yTest = train_test_split(df_transf[i]['X'], df_transf[i]['y'], test_size=0.3, random_state=0)
    Xgboost_calibrado_predictores[i] = XGBRegressor(random_state=0,learning_rate=0.35, max_depth=6,n_estimators=162).fit(XTrain,yTrain)
    print('Modelo de partición ' + str(i) + ' Progreso: ' + str(round(s/len(Particiones)*100,0)) + '%')
    print('Dim Xtrain: ' + str(XTrain.shape)+ 'Dim Xtrain: ' + str(yTrain.shape)+ 'Dim Xtrain: ' + str(XTest.shape)+ 'Dim Xtrain: ' + str(yTest.shape))
    
    yPred = Xgboost_calibrado_predictores[i].predict(XTest)
    yTest_Completo = np.concatenate((yTest_Completo,yTest.to_numpy()))
    yPred_Completo = np.concatenate((yPred_Completo,yPred))
    print('Resultados: ')
    Xgboost_calibrado_metricas[i]=Metricas(yTest,yPred)
    #Desempeño modelo: 
print('Resultados finales Concatenados:')
Xgboost_Calibrado_metricas_Total=Metricas(yTest_Completo,yPred_Completo)

pd.DataFrame(Xgboost_calibrado_metricas).T.rename(columns={0:'R2',1:'MSE',2:'MAE',3:'MAPE'})

"""#### Comparación de modelos"""

Nombres_Modelos= ['Random_Forest','XGBoost_Sin_Calibrar','XGBoost_Calibrado']

Data_Comparacion = pd.DataFrame([Random_Forest_metricas_Total,Xgboost_metricas_Total,Xgboost_Calibrado_metricas_Total]
             ,index=Nombres_Modelos,columns=['R2','MSE','MAE','MAPE'])
Data_Comparacion

np.arange(len(Data_Comparacion.columns)-1)

for k in np.arange(len(Data_Comparacion.columns)-1):
    print(Data_Comparacion.index.tolist()[k])

for i in Data_Comparacion.columns:
    plt.bar(x=Data_Comparacion.index,height=Data_Comparacion[i])
    for k in np.arange(len(Data_Comparacion.columns)-1):
        plt.text(x=Data_Comparacion.index.tolist()[k],y=Data_Comparacion[i][k],s=Data_Comparacion[i][k])
    plt.title(i)
    plt.show()

"""En base a los resultados presentados, podemos hacer las siguientes conclusiones:

R2: El coeficiente de determinación (R2) es un indicador de qué tan bien se ajusta el modelo a los datos. Cuanto más cercano a 1 sea el valor de R2, mejor se ajustará el modelo a los datos. En este caso, los modelos de Random Forest y XGBoost (calibrado y sin calibrar) tienen valores de R2 muy similares, lo que indica que todos los modelos se ajustan bien a los datos.

MSE: El error cuadrático medio (MSE) mide el promedio de los errores al cuadrado de las predicciones del modelo. Cuanto menor sea el valor de MSE, mejor será el modelo. En este caso, el modelo XGBoost calibrado tiene el valor más bajo de MSE, lo que indica que es el mejor modelo en términos de precisión.

MAE: El error absoluto medio (MAE) mide el promedio de los errores absolutos de las predicciones del modelo. Cuanto menor sea el valor de MAE, mejor será el modelo. En este caso, el modelo XGBoost calibrado tiene el valor más bajo de MAE, lo que indica que es el mejor modelo en términos de precisión.

MAPE: El error porcentual absoluto medio (MAPE) mide el promedio de los errores porcentuales de las predicciones del modelo. Cuanto menor sea el valor de MAPE, mejor será el modelo. En este caso, el modelo XGBoost calibrado tiene el valor más bajo de MAPE, lo que indica que es el mejor modelo en términos de precisión.

En general, podemos concluir que el modelo XGBoost calibrado es el mejor modelo para predecir el precio de los vehículos, ya que tiene los valores más bajos de MSE, MAE y MAPE. Sin embargo, los modelos de Random Forest y XGBoost sin calibrar también son modelos sólidos que se ajustan bien a los datos.

### Validación del modelo para Competencia
"""

print('Forma del dataTesting: ' + str(dataTesting.shape))
dataTesting_ID = dataTesting.reset_index()
dataTesting_ID.head()

Info_Make_models_test = dataTesting.apply(lambda x: x['Make']+x['Model'], axis=1).drop_duplicates().tolist()
for i in Info_Make_models_test:
    if i not in Info_Make_models:
        print('ERROR GRAVE!: ' + i)

# Visualización datos dae test
df_models_test = {}
suma=0
for i in Particiones:
    df_aux=pd.merge(Particiones_Marcas[i],dataTesting_ID,on=['Make','Model'])
    df_aux['Make-Mod'] = df_aux.apply(lambda x: x['Make']+x['Model'], axis=1)
    df_aux=df_aux.drop(columns=['Make','Model'],axis=1)
    df_models_test[i]=df_aux
    print('La cantidad de registros de la partición ' + str(i) + ' es ' + str(df_aux.shape))
    suma += df_aux.shape[0]
print('Cantidad total de registros: ' +str (suma))
df_models_test[0.2].head(5)

#Escalado de variables
Variable_ID = ['ID']

df_scaled_test = {}
Variable_y = ['Price']
Variables_numericas=['Year','Mileage']
Variables_categoricas= ['State','Make-Mod']
for i in Particiones:
    df_model_nums = df_models_test[i][Variables_numericas]
    scaled_data = scalers[i].transform(df_model_nums)
    scaled_data = pd.DataFrame(scaled_data,columns=['Year','Mileage'])
    df_scaled_test[i]=pd.concat([df_models_test[i][Variable_ID+Variables_categoricas],scaled_data],axis=1)
df_scaled_test[0.2].head()

#Codificación One-Hot
df_dummies_test = {}
for i in Particiones:
    df_dummies_test[i]=pd.get_dummies(df_scaled_test[i],columns=Variables_categoricas)
    print('La partición ' + str(i) + ' tiene la siguiente forma: ' + str(df_dummies_test[i].shape))
df_dummies_test[0.2]

A_Predecir = {}
for i in Particiones:
    Clave = {}
    Clave['xVal']= df_dummies_test[i].drop('ID',axis=1)
    Clave['id'] = df_dummies_test[i]['ID']
    A_Predecir[i]=Clave
A_Predecir[0.2]['id']

#Revisión de coherencia de columnas
for i in Particiones:
    Columnas_Val = A_Predecir[i]['xVal'].columns.tolist()
    Columnas_Train = df_transf[i]['X'].columns.tolist()
    for k in Columnas_Train:
        if k not in Columnas_Val:
            A_Predecir[i]['xVal'][k]=0
    A_Predecir[i]['xVal'] = A_Predecir[i]['xVal'].reindex(columns=Columnas_Train)
    Columnas_Val = A_Predecir[i]['xVal'].columns.tolist()
    Columnas_Train = df_transf[i]['X'].columns.tolist()
    print('Las columnas en la partición ' +str(i) +' son las mismas de entrenamiento: '+str(Columnas_Val==Columnas_Train))

#Predicción
Predicciones={}
for i in Particiones:
    yPred= Xgboost_calibrado_predictores[i].predict(A_Predecir[i]['xVal'])
    yIndex = A_Predecir[i]['id']
    Predicciones[i]=pd.DataFrame(zip(yIndex,yPred),columns=['ID','Price'])
Prediccion_Para_Subir = pd.concat(Predicciones.values()).sort_values('ID').set_index('ID')
Prediccion_Para_Subir.to_csv('test_submission.csv', index_label='ID')
Prediccion_Para_Subir.head()

"""#### Función para Predecir"""

def Api_para_Predecir(Diccionario_con_datos):
    df_api = pd.DataFrame(Diccionario_con_datos)
    #Transoormación
    df_api = df_api.reset_index().rename(columns={'index':'ID'})
    df_models_test = {}
    Particiones_api=[]
    suma=0
    for i in Particiones:
        df_aux=pd.merge(Particiones_Marcas[i],df_api,on=['Make','Model'])
        if df_aux.shape[0] > 0 :
            #print(i)
            Particiones_api.append(i)
            df_aux['Make-Mod'] = df_aux.apply(lambda x: x['Make']+x['Model'], axis=1)
            df_aux=df_aux.drop(columns=['Make','Model'],axis=1)
            df_models_test[i]=df_aux
            suma += df_aux.shape[0]
    #Escalado
    Variable_ID = ['ID']
    df_scaled_test = {}
    Variable_y = ['Price']
    Variables_numericas=['Year','Mileage']
    Variables_categoricas= ['State','Make-Mod']
    for i in Particiones_api:
        df_model_nums = df_models_test[i][Variables_numericas]
        scaled_data = scalers[i].transform(df_model_nums.to_numpy())
        scaled_data = pd.DataFrame(scaled_data,columns=['Year','Mileage'])
        df_scaled_test[i]=pd.concat([df_models_test[i][Variable_ID+Variables_categoricas],scaled_data],axis=1)

    #Codificación ONE-HOT
    df_dummies_test = {}
    for i in Particiones_api:
        df_dummies_test[i]=pd.get_dummies(df_scaled_test[i],columns=Variables_categoricas)

    A_Predecir = {}
    for i in Particiones_api:
        Clave = {}
        Clave['xVal']= df_dummies_test[i].drop('ID',axis=1)
        Clave['id'] = df_dummies_test[i]['ID']
        A_Predecir[i]=Clave

    #Revisión de coherencia de columnas
    for i in Particiones_api:
        Columnas_Val = A_Predecir[i]['xVal'].columns.tolist()
        Columnas_Train = df_transf[i]['X'].columns.tolist()
        for k in Columnas_Train:
            if k not in Columnas_Val:
                A_Predecir[i]['xVal'][k]=0
        A_Predecir[i]['xVal'] = A_Predecir[i]['xVal'].reindex(columns=Columnas_Train)
        Columnas_Val = A_Predecir[i]['xVal'].columns.tolist()
        Columnas_Train = df_transf[i]['X'].columns.tolist()
    Predicciones={}
    for i in Particiones_api:
        yPred= Xgboost_calibrado_predictores[i].predict(A_Predecir[i]['xVal'])
        yIndex = A_Predecir[i]['id']
        Predicciones[i]=pd.DataFrame(zip(yIndex,yPred),columns=['ID','Price'])
    Prediccion_Para_Subir = pd.concat(Predicciones.values()).sort_values('ID').set_index('ID')

    return Prediccion_Para_Subir['Price']

ejemplo = {'Year':[2015],
                   'Mileage':[43303],
                   'State':['CA'],
                   'Make':['Subaru'],
                   'Model':['Outback2.5i']}
Api_para_Predecir(ejemplo)