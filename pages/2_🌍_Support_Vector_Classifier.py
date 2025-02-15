import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError

import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Support Vector Classifier", page_icon="🌍")

st.markdown("# Support Vector Classifier")

st.write(
    """This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
to display geospatial data."""
)

st.sidebar.header("Support Vector Classifier")
