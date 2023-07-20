import streamlit as st
import process_image as process
from Vanila_Unet_model import *
from PIL import Image as im
import numpy as np
unet = Vanila_Unet()
model = unet.model_gen()
st.title('STEEL DEFECT DETECTION APPLICATION')
st.markdown("***")

st.subheader("Upload the image of the steel's surface")
option = st.radio('',('Single image', 'Multiple image'))
st.write('You selected:', option)

if option == 'Single image':
    uploaded_file = st.file_uploader(' ',accept_multiple_files = False)
    if uploaded_file is not None:
        pred_mask = process.predict(uploaded_file.name, model, False)
        st.image(uploaded_file)
        st.image(pred_mask)

elif option == 'Multiple image':
    uploaded_file = st.file_uploader(' ',accept_multiple_files = True)
    if len(uploaded_file) != 0:
        st.write("Images Uploaded Successfully")
        # Perform your Manupilations (In my Case applying Filters)
        for i in range(len(uploaded_file)):
            pred_mask = process.predict(uploaded_file[i].name, model, False)
            st.image(uploaded_file[i])
            st.image(pred_mask)
            
else:
    st.write("Make sure you image is in TIF/JPG/PNG Format.")