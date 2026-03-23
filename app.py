import streamlit as st

st.title("counterfactual explanation website")

x = st.slider("input number ", 0, 100)

st.write("number is :", x)