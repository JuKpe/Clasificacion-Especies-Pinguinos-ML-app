import streamlit as st

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Clasificación de Pingüinos 🐧 con Modelos de Machine Learning")
st.write(
    """Bienvenido a esta aplicación interactiva de clasificación de pingüinos. Aquí exploraremos 
    cómo tres poderosos modelos de aprendizaje automático pueden predecir la especie de un pingüino 
    en función de sus características físicas."""
)

#st.markdown("# hello juan")
st.sidebar.header("hola")


st.sidebar.success("Seleccione un modelo.")

st.markdown("""
Bienvenido a esta aplicación interactiva de clasificación de pingüinos.  
Aquí exploraremos cómo tres poderosos modelos de aprendizaje automático pueden predecir la especie de un pingüino en función de sus características físicas.  

### 📊 Modelos utilizados:
- 🌲 **Random Forest Classifier**
- 📈 **Support Vector Classifier (SVC)**
- 🔍 **K-Nearest Neighbors (KNN)**

Usando un conjunto de datos real sobre pingüinos, esta aplicación te permitirá comparar el rendimiento de estos modelos y visualizar sus predicciones de manera sencilla. 🚀  

💡 **Explora, ajusta los parámetros y descubre cuál modelo clasifica mejor a nuestros amigos emplumados!** 🐧✨
""")

st.markdown(
    """
    ### Nota:
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)

