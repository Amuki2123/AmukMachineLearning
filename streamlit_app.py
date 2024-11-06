import streamlit as st
import pandas as pd

st.title('🎈 Machine Learning App')

With st.expander('Data'):
st.write('**Raw data**')
st.info('This is my first Machine Learning App on Streamlit!')
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
df
