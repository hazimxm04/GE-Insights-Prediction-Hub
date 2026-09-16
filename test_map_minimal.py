import streamlit as st
import plotly.express as px
import pandas as pd

st.title("Basic Chart Test")

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
fig = px.bar(df, x='x', y='y')

fig.write_html("test_map.html") 
st.plotly_chart(fig, use_container_width=True)

st.write("If you see a bar chart above, Plotly rendering works at all.")