import time
import json
import os
import numpy as np
import pandas as pd
from math import ceil
import cv2
import tensorflow as tf
from flask import Flask, flash, request, redirect, url_for, render_template, Session
from werkzeug.utils import secure_filename
import numpy
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import keras
from keras.backend import clear_session
from keras.models import load_model
from keras.optimizers import SGD,RMSprop
from keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import jsonify
from tensorflow.compat.v1 import set_random_seed
SEED = 7
np.random.seed(SEED)
set_random_seed(SEED)

graph = tf.get_default_graph()
session = tf.Session(graph = tf.Graph())
with session.graph.as_default():
    keras.backend.set_session(session)
    classifier = load_model('resnet_2048.h5')
    classifier.load_weights('resnet_2048_weights.h5', by_name=True)
    classifier._make_predict_function()
#Pre-processing
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim ==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
        
def circle_crop(img):   
     
    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))  
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    return img 
        
def load_ben_color(image, sigmaX=10):
    #image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (IMG_DIM, IMG_DIM))
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_DIM, IMG_DIM))
    image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    image = circle_crop(image)  
    return image
    
IMG_DIM = 256

#Load Image
def D(img_path):        
    df_class = pd.read_csv("uploader/2.csv")
    #df_class.id_code = df_class.id_code.apply(lambda x: x + ".png")
    df_class['id_code'] = df_class['id_code'].astype('str')
    #classifier.summary()
    #Single Prediction        
    test_img = cv2.imread(img_path)
    test_img = cv2.resize(test_img,(IMG_DIM,IMG_DIM))
    test_img = np.expand_dims(test_img, axis=0) 
    dummy_datagen=ImageDataGenerator(rescale=1./255, preprocessing_function=load_ben_color)
    #dummy_generator = dummy_datagen.flow(test_img, y=None, batch_size=1, seed=7)
    dummy_generator = dummy_datagen.flow_from_dataframe(dataframe=df_class,
                                                #directory="../input/aptos2019-blindness-detection/train_images/",
                                                directory="uploader/",                                                    
                                                x_col="id_code",
                                                #y_col="diagnosis",
                                                #y_col=["diagnosis_0","diagnosis_1","diagnosis_2"],
                                                batch_size=1,
                                                class_mode=None,
                                                target_size=(IMG_DIM, IMG_DIM),
                                                #shuffle=False,
                                            )
    with session.graph.as_default():
        keras.backend.set_session(session)
        tta_steps = 5
        preds_tta = []
        for i in range(tta_steps):
            dummy_generator.reset()
            preds = classifier.predict_generator(generator=dummy_generator, steps=ceil(df_class.shape[0]))
            preds_tta.append(preds)
        final_pred = np.mean(preds_tta, axis=0)
        predicted_class_indices = np.argmax(final_pred, axis=1)

        #Label Dictionary
        label_maps = {0: 'No DR', 1: 'Non-Proliferative DR', 2: 'Proliferative DR'}
        label = label_maps[int(predicted_class_indices)]
        return (label)


i=1
j=2
UPLOAD_FOLDER = 'uploader/'

app = Flask(__name__)
app.secret_key = 'some_secret'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/Info")
def Info():
		return render_template('Info.html')

@app.route("/DR", methods=['GET', 'POST'])
def DR():
		return render_template('base.html')
        
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('base.html')
    else:   
        file = request.files['image']
        file.save(os.path.join('uploader', "2.png"))
        res = D('uploader/2.png') 
        return str(res)
    return None
    



if __name__ == "__main__":
	app.run(host='127.0.0.1', port=5001, debug=True)
