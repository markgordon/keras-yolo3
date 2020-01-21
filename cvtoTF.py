import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model( 'model_data/ker.h5',
                                         input_layer={'embedding_input': [0,384,288,3]})
converter.experimental_new_converter = true
tfmodel = converter.convert()
open ("yolo_prn.tflite" , "wb") .write(tfmodel)