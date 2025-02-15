import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError

import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Support Vector Classifier", page_icon="🌍")

st.title("Support Vector Classifier")
st.markdown("""
SVC es un modelo basado en encontrar el hiperplano óptimo que separa las clases en el espacio de características.  
Funciona bien en conjuntos de datos con clases bien separadas.  
✅ **Ventajas:** Eficiente en espacios de alta dimensión y con distintos núcleos para adaptación.  
⚠️ **Desventaja:** Puede ser sensible al ajuste de hiperparámetros y a datos ruidosos.  
""")


st.sidebar.header("Hola")
st.sidebar.success("Seleccione un modelo.")
