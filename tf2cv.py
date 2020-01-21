#! /usr/bin/env python

import tensorflow as tf
from numpy import loadtxt
from tensorflow.keras.models import load_model
import argparse
import configparser
import io
import os
from collections import defaultdict

parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('input_path', help='Path to Darknet cfg file.')
parser.add_argument('output_path', help='Path to Darknet weights file.')
# load model
# %%
def _main(args):

     input_path = os.path.expanduser(args.input_path)
     output_path = os.path.expanduser(args.output_path)
     model = load_model(input_path)
# Convert the model.
     converter = tf.lite.TFLiteConverter.from_keras_model(model)
     tflite_model = converter.convert()
     open (output_path , "wb") .write(tflite_model)

if __name__ == '__main__':
    _main(parser.parse_args())