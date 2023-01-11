#!/usr/bin/env python
# coding: utf-8

from torch.jit.annotations import Optional
from fastai.vision.all import *
# from fastai.vision.all import open_image, load_learner, image, torch
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import time
import PIL.Image
import requests
from io import BytesIO
import pprint
import pandas as pd
#fetch the image from the URL

# App title
st.title("Fossil classifier")

def fetch_image(url):
    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    return img

def display_image(url):
    response = requests.get(url)
    pil_img = PIL.Image.open(BytesIO(response.content))
    img_disp = np.asarray(pil_img)
    return img_disp

def predict(url):    
    # Display the test image
    img_dispp = display_image(url)
    st.image(img_dispp, width=500)
            
    # Temporarily displays a message while executing 
    with st.spinner('Wait for it...Predicting...'):
        time.sleep(3)

    #Fetch image from url
    img = fetch_image(url)

    #model = load_learner('model/modelfile/')
    model = load_learner('model/modelfile/', 'model.pkl')
    pred_class,pred_idx,outputs = model.predict(img)
    res =  zip(model.data.classes, outputs.tolist())
    predictions = sorted(res, key=lambda x:x[1], reverse=True)
    top_predictions = predictions[0:5]
    df = pd.DataFrame(top_predictions, columns =["Fossil","Probibility"])
    df['Probibility'] = df['Probibility']*100
    st.write(df)
    return img
    
def predict_img(img_test):
    # Temporarily displays a message while executing 
    with st.spinner('Wait for it...Predicting...'):
        time.sleep(3)

    #model = load_learner('model/modelfile/')
    model = load_learner('model/modelfile/', 'model.pkl')
    #model = load_learner('C:\\Users\\H231148\\OneDrive - Halliburton\\Desktop\\models','model.pkl')
    pred_class,pred_idx,outputs = model.predict(img_test)
    res =  zip(model.data.classes, outputs.tolist())
    predictions = sorted(res, key=lambda x:x[1], reverse=True)
    top_predictions = predictions[0:5]
    df = pd.DataFrame(top_predictions, columns =["Fossil","Probability"])
    df['Probability'] = df['Probability']*100
    st.write(df)

# Image source selection
#option = st.radio('', ['Choose a test image', 'Choose your own image'])
option = st.radio('', ['Choose a sample image','Choose your own image'])

if option == 'Choose a sample image':


    # Test image selection
    test_images = os.listdir('data/test/')
    #test_images = os.listdir('C:\\Users\\H231148\\OneDrive - Halliburton\\Desktop\\test\\')
    test_image = st.selectbox('Please select a test image:', test_images)

    # Read the image
    file_path = 'data/test/' + test_image
    #file_path = 'C:\\Users\\H231148\\OneDrive - Halliburton\\Desktop\\test\\' + test_image
    img = open_image(file_path)
    # Get the image to display
    display_img = mpimg.imread(file_path)

    st.image(display_img, width=500)

    # Predict and display the image
    predict_img(img)

elif option == 'Choose your own image':
    url = st.text_input("Please input a url:")

    if url != "":
        #predict(url)
        try:

            #Predict and display the image
            predict(url)
        except:
            st.text("Invalid url!...Try some other url (eg: https://upload.wikimedia.org/wikipedia/commons/5/5c/Ammoniteplit.jpg)")

