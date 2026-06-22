# Predict adsoprtion capacity for three polymers!

A simple Streamlit app allows you to predict logKd for three different polymers (PE, PP, PS).

You can choose between two models:
- **Gaussian-based model**: Descriptors used by the model were calculated by Gaussian (π, M)
- **Rdkit-based model**: Descriptors used by the model were calculated by Rdkit (π, M)

### How to run it?

#### Just click in the link below:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mp-adsorbnet.streamlit.app)

### What do you need?
For every organic compound you will need:
- **logD**: n-octanol/water distribution coefficient at special pH value
- **π**: ratio of average molecular polarizability and molecular volume
- **M**: molecular mass
