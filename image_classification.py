import numpy as np
import streamlit as st
from keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from PIL import Image, ImageOps

def predict(image_path):
    #Load the Model from Json File
        json_file = open('modelVGG_pretrained.json', 'r')
        model_json_c = json_file.read()
        json_file.close()
        model_c = model_from_json(model_json_c)
        #Load the weights
        model_c.load_weights("vgg16_tf_cat_dog_final_dense.h5")
        #Compile the model
        opt = SGD(lr=1e-4, momentum=0.9)
        model_c.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
        #load the image you want to classify
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = image_path
        #image sizing
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)        
            #predict the image
        preds = model_c.predict(np.expand_dims(image, axis=0))[0]
        if preds==0:
            st.write("Predicted Label:Cat")
        else:
            st.write("Predicted Label:")
