import streamlit as st
from ucimlrepo import fetch_ucirepo, list_available_datasets

st.set_page_config(page_title="Support Vector Classifier", page_icon="🔍")

st.title("k nearest neighbor (KNN)")

st.markdown("""
KNN clasifica los datos según la mayoría de sus vecinos más cercanos. Es un modelo basado en la similitud.  
Cuanto menor sea la distancia entre puntos, mayor será la probabilidad de que pertenezcan a la misma clase.  
✅ **Ventajas:** Simple, no requiere entrenamiento extenso y se adapta bien a datos no lineales.  
⚠️ **Desventaja:** Puede ser lento en grandes conjuntos de datos y sensible a valores atípicos.  
""")

# check which datasets can be imported
list_available_datasets()

# import dataset
heart_disease = fetch_ucirepo(id=45)
# alternatively: fetch_ucirepo(name='Heart Disease')

# access data
X = heart_disease.data.features
y = heart_disease.data.targets

X
