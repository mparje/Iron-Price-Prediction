import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import set_option
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import GridSearchCV

def process_data(input_file, delete_nan=False, train=False):
    data = pd.read_csv(input_file, skipinitialspace=True)
    # Resto del código de la función process_data
    
st.title("Aplicación de predicción de precios de hierro")
st.write("Esta aplicación utiliza modelos de regresión para predecir los precios de hierro.")

uploaded_file = st.file_uploader("Cargar archivo CSV", type="csv")
if uploaded_file is not None:
    data, instance_id = process_data(uploaded_file.name)
    
    st.success("Los datos se han procesado correctamente.")

st.header("Datos procesados")
if "data" in locals():
    st.write(data)

st.header("División de datos")
if "data" in locals():
    y = data['Price-in-dollars']
    X = data.drop('Price-in-dollars', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    st.write("Número de datos de entrenamiento:", len(X_train))
    st.write("Número de datos de prueba:", len(X_test))
    
st.header("Configuración del modelo")
if "data" in locals():
    param_dist = {'n_estimators': [10, 20, 25, 50, 75],
                  'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 14],
                  'reg_lambda': [0.26, 0.25, 0.2, 0.3]}
    
    rf = xgboost.XGBRFRegressor()
    
    rf_cv = RandomizedSearchCV(rf, param_distributions=param_dist, random_state=0, n_jobs=-1)
    
    rf_cv.fit(X, y)
    
    st.write("Parámetros ajustados del modelo:", rf_cv.best_params_)
    
st.header("Predicción de precios")
if "data" in locals():
    Ran = xgboost.XGBRFRegressor(reg_lambda=0.26, n_estimators=75, max_depth=9, random_state=0)
    Ran.fit(X_train, y_train)
    main_prediction = Ran.predict(data_test)
    
    data_predicted = pd.DataFrame()
    data_predicted['ID'] = instance_id
    data_predicted['Price-in-dollars'] = main_prediction
    
    st.write("Número total de predicciones:", len(data_predicted))
    
    st.subheader("Guardar predicciones")
    if st.button("Guardar predicciones en CSV"):
        data_predicted.to_csv('Predictions/price_prediction.csv', index=False)
        st.success("Las predicciones se han guardado correctamente.")

    st.subheader("Evaluación del modelo")
    y_pred = Ran.predict(X_test)
    errors = abs(y_pred - y_test)
    mape = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mape)
    st.write("Exactitud del modelo:", accuracy)
