import streamlit as st
from Vanila_Unet_model import *
from PIL import Image as im
import numpy as np
import glob
import cv2
import os

def binarize_custom(masks, th = 0.1):
    # Maximum value of each channel per pixel
    m = masks
    # Binarization
    m = (m>th) * 255
    return m

def predict(img, model, show_img = False):
    # name = path.split('/')[-1]
    img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_gray is None:
        print("Cant read image")
    else:
        img_gray = cv2.resize(img_gray, (1600, 256))
        img_ = img_gray[..., np.newaxis]    # Add channel axis
        img_ = img_[np.newaxis, ...]    # Add batch axis
        img_ = img_ / 255.              # 0～1
        
        masks = model.predict(img_)
        pred_mask = masks[0,:,:,0]
        for i in range(1,4):
            pred_mask +=  masks[0,:,:,i]
        pred_mask = binarize_custom(pred_mask, 0.1)
        if show_img:
            img = cv2.imread(path)
            return img, pred_mask
        else: 
            return pred_mask
        

model = None
if model is None:
    unet = Vanila_Unet()
    model = unet.model_gen()
st.title('STEEL DEFECT DETECTION APPLICATION')
st.markdown("***")

st.subheader("Upload the image of the steel's surface")
option = st.radio('',('Single image', 'Multiple image'))
st.write('You selected:', option)

if option == 'Single image':
    uploaded_file = st.file_uploader(' ',accept_multiple_files = False, label_visibility = "hidden")

    if uploaded_file is not None:   
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        pred_mask = predict(opencv_image, model, False)
        print(uploaded_file.name)
        st.image(uploaded_file)
        st.image(pred_mask)

elif option == 'Multiple image':
    uploaded_file = st.file_uploader(' ',accept_multiple_files = True)
    if uploaded_file is not None:   
        if len(uploaded_file) != 0:
            st.write("Images Uploaded Successfully")
            # Perform your Manupilations (In my Case applying Filters)
            for i in range(len(uploaded_file)):
                file_bytes = np.asarray(bytearray(uploaded_file[i].read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                
                pred_mask = predict(opencv_image, model, False)
                st.image(uploaded_file[i])
                st.image(pred_mask)
            
else:
    st.write("Make sure you image is in TIF/JPG/PNG Format.")