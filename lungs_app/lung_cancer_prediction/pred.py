import tensorflow as tf
import pandas as pd   
import numpy as np
import cv2
import os
import efficientnet.tfkeras as efn
import glob
from django.conf import settings
import os
import uuid

#load model
def load_model(modl):
    return tf.keras.models.load_model(modl, compile=False)


def __predict_one_image(img: np.ndarray, model) -> np.ndarray:
    # return the predictions
    return np.squeeze(model.predict(img)[0])

def __compute_confidence(preds: np.ndarray,num_class:int) -> float:
    # compute the confidence according to the problem statement
    if num_class == 2:
        # compute the confidence according to the prediction
        if preds < 0.5:
            conf = 1 - preds
        else:
            conf = preds
    else:
        conf = np.amax(preds)  #amax returns the maimum of an array

    # return the confidence score
    return conf

def __threshold_predictions(preds: np.ndarray,num_class:int) -> int:
    # copy the predictions
    p = preds.copy()

    # threshold the predictions, according to the problem
    if num_class == 2:
        p[p < 0.5] = 0
        p[p != 0] = 1
    else:
        p = np.argmax(p, axis=-1)

    # return the thresholded predictions
    return p.astype("int")


#load the images and preprocess the images
def preprocess_img(img):
    img = cv2.imread(img)
    img = cv2.resize(img,(224,224))
    # convert the image to a tensor for inference
    return np.expand_dims(img, 0).astype(np.float32) / 255.0

def show(img, conf, name):
    img_name = img.split('/')[-1].split('.')[0]
    print(img_name)
    img = cv2.imread(img)
    cv2.putText(img,name,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(img,'conf = ' + str(conf) ,(60,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    
    unique_filename = str(uuid.uuid4().hex) + ".jpg"
    print("image name = = ",unique_filename)
    path = os.path.join(settings.MEDIA_ROOT, "trained", unique_filename)
    cv2.imwrite(path,img)
    # cv2.imshow(img_name,img)
    print("path = = ", path)
    return path


def startloadmodel(uploaded_img):
    #load the model
    clas = ['Bengin', 'Malignant', 'Normal']
    num_class = len(clas)
    model = load_model(os.path.join(settings.BASE_DIR, "model", "best.h5"))
    print("model: ",model)

    #multiple image 
    # for image in glob.glob("C:/Users/bhuwa/OneDrive/Desktop/2022 Projects/LungsProject/lung_cancer_prediction/testing_dataset/*.jpg"):
    #     print("image:",image)
    #     img  =  preprocess_img(image)
    #     preds = __predict_one_image(img)
    #     print("preds: ",preds)
    #     print("max preds: ",np.argmax(preds))
    #     conf = __compute_confidence(preds,num_class)
    #     print("confidence score: ",conf)
    #     lbl = __threshold_predictions(preds,num_class)
    #     print("class name :", clas[lbl])

    #show image and confidence and labelname
    # show(image,conf,clas[lbl])

#     #single image 

    image = uploaded_img
    img  =  preprocess_img(image)
    preds = __predict_one_image(img, model)
    print("preds: ",preds)
    print("max preds: ",np.argmax(preds))
    conf = __compute_confidence(preds,num_class)
    print("confidence score: ",conf)
    lbl = __threshold_predictions(preds,num_class)
    print("class name :", clas[lbl])

# #show image and confidence and labelname
    path = show(image,conf,clas[lbl])

    output_img = path
    print("output = ", output_img)
    return conf, output_img, clas[lbl]

