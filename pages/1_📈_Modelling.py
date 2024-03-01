"""Second page of the App"""
import streamlit as st

st.header("Machine and Deep Learning Models")
st.subheader("Experimentation Results")

tabs = st.tabs(["Machine Learning", "Deep Learning"])

with tabs[0]:
    st.write("Machine Learning")
with tabs[1]:
    st.write("Deep Learning")
