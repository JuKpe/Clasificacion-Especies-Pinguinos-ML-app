import streamlit as st
from ucimlrepo import fetch_ucirepo, list_available_datasets

st.write("# Welcome to prueba! 👋")

# check which datasets can be imported
list_available_datasets()

# import dataset
heart_disease = fetch_ucirepo(id=45)
# alternatively: fetch_ucirepo(name='Heart Disease')

# access data
X = heart_disease.data.features
y = heart_disease.data.targets

X
