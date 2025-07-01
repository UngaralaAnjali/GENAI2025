import streamlit as st
from PIL import Image
img = Image.open("c:/Users/DELL/Pictures/Screenshots/Screenshot (2).png")
st.image(img, caption="Sample Image")
