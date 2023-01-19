import streamlit as st
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import io
from datetime import date
import csv

st.set_page_config(page_title='iEat')
st.markdown("""
    <h1 style='text-align: center; color: #FFFFFF; margin-bottom: -30px;'>
  iEat
    </h1>
""", unsafe_allow_html=True
)
st.caption("""
    <p style='text-align: center; color: #FFFFFF;'>
    by <a href='https://www.rikeshpatel.io/'>Rikesh Patel</a>
    </p>
""", unsafe_allow_html=True
)

tab1,tab2 = st.tabs(["Main", "Nutrition Facts"])

with tab1:

   st.write("#### Just snap a picture of your plate!")

   image = None


   # Image input for upload of dish
   image = st.file_uploader("Your food", ["png", "jpg", "jpeg", 'HEIC'], key = 'file')


   # Load and compile the Model
   model = load_model('vgg19.h', compile = False)
   loss = 'sparse_categorical_crossentropy'
   optimizer = 'adam'
   model.compile(loss = loss, optimizer = optimizer)

   # Get helper information
   list = open('classes.txt')
   classes = list.read()
   classes = classes.split(sep = '\n')
   classes = classes[:-1]
   list.close()

   nutrients = pd.read_csv('nutrient_facts.csv', index_col = 0, encoding='latin-1')
   history= pd.read_csv('food.csv', index_col = 0)


   
        
   def make_prediction(img, model = model):
      '''
       This function takes in a image and model, and uses the model to predict the class of the image 
      '''
      img = img.convert('RGB')
      img = img.resize((224,224), Image.NEAREST)
      img = tf.keras.preprocessing.image.img_to_array(img)
      img_array = tf.expand_dims(img, 0) 
      prediction = model.predict(img_array) 
      pred_class = np.argmax(prediction) 
      return classes[pred_class]


   if image is not None: 
       col1, col2 = st.columns(2) 
       with col1: 
           image_to_share = Image.open(image)
           st.image(image_to_share, width=265)
       with col2: 
            st.write("## The predicted class is")
            predicted_class = make_prediction(image_to_share)
            st.write('# {}'.format(predicted_class))
            food = nutrients.loc[predicted_class].T
            st.write(food)
            image = None

           



with tab2:
    st.write("### Select a food for facts!")
    
    nutrients = pd.read_csv('nutrient_facts.csv', index_col = 0, encoding='latin-1')
    

    selected = st.multiselect("Select a food.", nutrients.index.tolist())
    
    
    if st.button("Submit"):
        st.write(nutrients.loc[selected])
        
