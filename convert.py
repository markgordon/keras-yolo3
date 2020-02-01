#! /usr/bin/env python
"""
Reads Darknet config and weights and creates Keras model with TF backend.

"""


import configparser
import argparse
import configparser
import io
import os
from collections import defaultdict
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (Conv2D, Input, ZeroPadding2D, Add,
                          UpSampling2D, MaxPooling2D, Concatenate)
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils.vis_utils import plot_model as plot
#from tensorflow.keras.utils.vis_utils import plot_model as plot
from tensorflow_model_optimization.python.core import sparsity
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from train import get_anchors,get_classes,data_generator_wrapper
import numpy as np
parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras model file.')
parser.add_argument(
    '-p',
    '--plot_model',
    help='Plot generated Keras model and save as image.',
    action='store_true')
parser.add_argument(
    '-w',
    '--weights_only',
    help='Save as Keras weights file instead of model file.',
    action='store_true')

def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream

# %%
def make_model(model_file, weights_file,anchor_file,**kwargs):
    annotation_path = 'model_data/combined1.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = np.load(anchor_file,allow_pickle=True)
    model_path = 'model_data/'
    init_model= model_path + '/pelee3'
    new_pruned_keras_file = model_path + 'pruned_' + init_model
    epochs = 100
    batch_size = 16
    init_epoch = 50
    input_shape = (384,288) # multiple of 32, hw
    log_dir = 'logs/000/'
    config_path = model_file
    weights_path = weights_file
    output_path = model_file + '.tf'
    output_root = os.path.splitext(output_path)[0]
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(
        shape=(3, ), dtype='int32', buffer=weights_file.read(12))
    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    print('Parsing Darknet config.')
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)
    first_layer = True
    print('Creating Keras model.')
    all_layers = []
    weight_decay = float(cfg_parser['net_0']['decay']
                         ) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0
    out_index = []
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)

            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn'
                  if batch_normalize else '  ', activation, weights_shape)

            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters

            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias
            ]

            # Handle activation.
            act_fn = None
            if activation != 'linear':
                pass  # Add advanced activation later.
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            # Create Conv2D layer
            if stride>1:
                # Darknet uses left and top padding instead of 'same' mode
                prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer)
            if(first_layer):
                conv_layer = Conv2D(
                    filters, (size, size),
                    strides=(stride, stride),
                    kernel_regularizer=l2(weight_decay),
                    use_bias=not batch_normalize,
                    weights=conv_weights,
                    activation=act_fn,
                    padding=padding)(prev_layer)
            else:
                conv_layer = prune.prune_low_magnitude(Conv2D(
                        filters, (size, size),
                        strides=(stride, stride),
                        kernel_regularizer=l2(weight_decay),
                        use_bias=not batch_normalize,
                        weights=conv_weights,
                        activation=act_fn,
                        padding=padding))(prev_layer)
            if batch_normalize:
                conv_layer = BatchNormalization(
                    weights=bn_weight_list)(conv_layer)
            prev_layer = conv_layer
            first_layer=False
            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
            elif activation == 'swish':
                act_layer = sigmoid(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                print('Concatenating route layers:', layers)
                concatenate_layer = Concatenate()(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer)
                prev_layer = skip_layer
			
        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    pool_size=(size, size),
                    strides=(stride, stride),
                    padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            all_layers.append(Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]
            all_layers.append(LeakyReLU(alpha=0.1)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            assert stride == 2, 'Only stride=2 supported.'
            all_layers.append(UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('yolo'):
            out_index.append(len(all_layers)-1)
            all_layers.append(None)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            height = int(cfg_parser[section]['height'])
            width = int(cfg_parser[section]['width'])
            input_layer = Input(shape=(height, width, 3))
            prev_layer = input_layer
            input_shape = (width, height)

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save model.
    if len(out_index)==0: out_index.append(len(all_layers)-1)
    num_anchors = len(anchors[1])
    if(len(out_index)>0):
        shape = K.int_shape(all_layers[out_index[0]])
        y1_reshape = Backend.reshape(all_layers[out_index[0]],(shape[1],shape[2], num_anchors, 5 + num_classes))
    if(len(out_index)>1):
        shape = K.int_shape(all_layers[out_index[1]])
        y2_reshape = Backend.reshape(all_layers[out_index[1]],(shape[1],shape[2], num_anchors, 5 + num_classes))
    yolo_model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
    yolo_model_wrapper = Model(input_layer, [y1_reshape, y2_reshape])
    print(yolo_model.summary())
    return yolo_model,yolo_model_wrapper

    if False:
        if args.weights_only:
            model.save_weights('{}'.format(output_path))
            print('Saved Keras weights to {}'.format(output_path))
        else:
            model.save('{}'.format(output_path),save_format='tf')
            print('Saved Keras model to {}'.format(output_path))

        # Check to see if all weights have been read.
        remaining_weights = len(weights_file.read()) / 4
        weights_file.close()
        print('Read {} of {} from Darknet weights.'.format(count, count +
                                                           remaining_weights))
        if remaining_weights > 0:
            print('Warning: {} unused weights'.format(remaining_weights))

    if True:
        model = create_model(model, anchors, num_classes, input_shape, input_layer, layers, out_index)
        yolo_model_wrapper.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer='adam',
            metrics=['accuracy'],
            callbacks = [
                sparsity.keras.pruning_callbacks.UpdatePruningStep(),
                sparsity.keras.pruning_callbacks.PruningSummaries(log_dir=log_dir, profile_batch=0)
            ]
            )
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')
        print(model.summary())

        batch_size = 16 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=5,
            initial_epoch=0)


       #m2train.m2train(args,model)
        #score = model.evaluate(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        #                       class_names, verbose=0)
        #print('Test loss:', score[0])
        #print('Test accuracy:', score[1])
    final_model=model
    final_model = sparsity.keras.prune.strip_pruning(model)
    final_model.summary()
    print('Saving pruned model to: ', new_pruned_keras_file)
    final_model.save('{}'.format(output_path),save_format='tf')
    tflite_model_file = model_path + "sparse.tf"
    converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    with open(tflite_model_file, 'wb') as f:
      f.write(tflite_model)

if __name__ == '__main__':
    _main(parser.parse_args())