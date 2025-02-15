import streamlit as st
from sklearn.neighbors import NearestCentroid
import pandas as pd
import numpy as np
#from ucimlrepo import fetch_ucirepo, list_available_datasets

st.set_page_config(page_title="Support Vector Classifier", page_icon="🔍")

st.title("k nearest neighbor (KNN)")

st.markdown("""
KNN clasifica los datos según la mayoría de sus vecinos más cercanos. Es un modelo basado en la similitud.  
Cuanto menor sea la distancia entre puntos, mayor será la probabilidad de que pertenezcan a la misma clase.  
✅ **Ventajas:** Simple, no requiere entrenamiento extenso y se adapta bien a datos no lineales.  
⚠️ **Desventaja:** Puede ser lento en grandes conjuntos de datos y sensible a valores atípicos.  
""")

# check which datasets can be imported
#list_available_datasets()

# import dataset
#heart_disease = fetch_ucirepo(id=45)
# alternatively: fetch_ucirepo(name='Heart Disease')

# access data
#X = heart_disease.data.features
#y = heart_disease.data.targets

#X

# ------------------------------------------------------------

with st.expander('DATA'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/JuKpe/data/refs/heads/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw

  st.write('**EDA**')
  st.write(df.describe())


with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Input features
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Sex', ('male', 'female'))

  # Create a DataFrame for the input features
  data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0) #combina los nuevos datos de entrada con las X

with st.expander ('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins

# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  st.write('Adelie: 0  -  Chinstrap: 1  -  Gentoo: 2')
  y

# Model training and inference
## Train the ML model
clf = NearestCentroid()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0])) # esto indica la posición de una matriz [fila][columna]

