import tensorflow as tf
from numpy import loadtxt
from tensorflow.keras.models import load_model
 
# load model
model = load_model('model_data/ker1.h5')
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open ("yolo_prn.tflite" , "wb") .write(tflite_model)
