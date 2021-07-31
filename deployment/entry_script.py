import os, json, requests
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

# Function executed when the model is deployed for the first time
def init():
    global model
    global model_folder
    global version
    
    version = '1' # Model version = name of the folder under 'TFSavedModel'
    
    model_folder = os.getenv('AZUREML_MODEL_DIR') # location of the model folder in Azure ML
    
    if (model_folder==None): # for testing in a local environment
        model_folder = ".." # location of TFSavedModel folder relative to this script
        
    model_path = os.path.join(model_folder, 'TFSavedModel', version) # full path of the model, including the version number (modify the version number if you saved it under an different number)

    model = tf.keras.models.load_model(model_path) # load the model into memory
    
# Function executed every time the model is called (inference)
@rawhttp
def run(request):

    newsize = (224, 224) # the picture uploaded by the user will be resized to this size before being passed to the model
    
    labels = pickle.load(open(os.path.join(model_folder, 'TFSavedModel', version, 'labels.pickle'), 'rb')) # load the labels (classes)
    
    if request.method == 'POST': # if the model is called by POST request
        file = request.files['file'] # get the file uploaded with the POST request
        img = Image.open(file) # open the file as a PIL image
        img = img.convert('RGB') # convert to RGB
        img = img.resize(newsize, Image.NEAREST) # resize the image
        img_array = image.img_to_array(img) # convert the image into an array
        img_batch = np.expand_dims(img_array, axis=0) # expand dimensions of the image
        img_preprocessed = preprocess_input(img_batch) # preprocess the image for the model

        prediction = model.predict(img_preprocessed) # call the model to get a prediction (= probabilities for each class)
        prediction = np.array(prediction[0])
        
        print(prediction)
        
        result = {} # dictionary that will contain the labels (classes) and their associated probability
        for index, probability in enumerate(prediction):
            result[labels[index]] = probability.astype(float)
    
        return result # return the dictionary as JSON format
        
    else:
        return AMLResponse("bad request, use POST", 500)    
    return 0
