import streamlit as st
import pandas as pd

st.title('🎈 Machine Learning App')
st.info('This is my first Machine Learning App on Streamlit!')

With st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df
  st.write('**x**')
  x = df.drop('species',axis=1)
  x
  st.write(**y**)
  y = df.species
  y
