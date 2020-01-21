#!/usr/bin/env python3

# In weiten Teilen aus https://www.curious-containers.cc/docs/machine-learning-guide entnommen (gesehen 18.01.2020)

import os
import sys
import argparse
import h5py
import statistics

from ensemble import Ensemble
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

# Constants
WEIGHTS_FILE = 'weights.h5'
METRICS_FILE = 'metrics.txt'
TRAIN_X_FILE = 'camelyonpatch_level_2_split_train_x.h5'
TRAIN_Y_FILE = 'camelyonpatch_level_2_split_train_y.h5'
VALID_X_FILE = 'camelyonpatch_level_2_split_train_x.h5'
VALID_Y_FILE = 'camelyonpatch_level_2_split_train_y.h5'
INPUT_SHAPE = (96, 96, 3)
NUM_CLASSES = 2


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    dest='data_dir', type=str,
    help='Data: Path to read-only directory containing PCAM *.h5 files.'
)
parser.add_argument(
    '--learning-rate', type=float, default=0.0005,
    help='Training: Learning rate. Default: 0.0005'
)
parser.add_argument(
    '--batch-size', type=int, default=128,
    help='Training: Batch size. Default: 128'
)
parser.add_argument(
    '--num-epochs', type=int, default=10,
    help='Training: Number of epochs. Default: 10'
)
parser.add_argument(
    '--steps-per-epoch', type=int, default=None,
    help='Training: Steps per epoch. Default: data_size / batch_size'
)
parser.add_argument(
    '--ensemble-count', type=int, default=10,
    help='Training: Networks in ensemble. Default: 10'
)
parser.add_argument(
    '--log-dir', type=str, default=None,
    help='Debug: Path to writable directory for a log file to be created. Default: log to stdout / stderr'
)
parser.add_argument(
    '--log-file-name', type=str, default='training.log',
    help='Debug: Name of the log file, generated when --log-dir is set. Default: training.log'
)
args = parser.parse_args()


# Redirect output streams for logging
if args.log_dir:
    log_file = open(os.path.join(os.path.expanduser(args.log_dir), args.log_file_name), 'w')
    sys.stdout = log_file
    sys.stderr = log_file

# Make the ensemble
networks = []
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
for i in range(args.ensemble_count):
    model_x = tf.keras.models.clone_model(model)
    model_x.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(args.learning_rate),
        metrics=['accuracy', AUC()]
    )
    networks.append(model_x)

ensemble = Ensemble(networks, NUM_CLASSES)

# Input
data_dir = os.path.expanduser(args.data_dir)

train_x = h5py.File(os.path.join(data_dir, TRAIN_X_FILE), 'r', libver='latest', swmr=True)['x']
train_y = h5py.File(os.path.join(data_dir, TRAIN_Y_FILE), 'r', libver='latest', swmr=True)['y']
valid_x = h5py.File(os.path.join(data_dir, VALID_X_FILE), 'r', libver='latest', swmr=True)['x']
valid_y = h5py.File(os.path.join(data_dir, VALID_Y_FILE), 'r', libver='latest', swmr=True)['y']

# Training
data_size = len(train_x)
steps_per_epoch = data_size // args.batch_size

if args.steps_per_epoch:
    steps_per_epoch = args.steps_per_epoch

ensemble.train(train_x, train_y, args.batch_size, steps_per_epoch, args.num_epochs, valid_x, valid_y)

# Output
accuracies = ensemble.getAccuracies(valid_x, valid_y)
erg = ensemble.getUncertaintyCorrectness(valid_x, valid_y)
f = open(METRICS_FILE, 'w')
f.write('All accuracies: ' + str(accuracies) + '\n')
f.write('Average accuracy: ' + str(statistics.mean(accuracies)) + '\n')
f.write('Ensemble accuracy: ' + str(ensemble.getEnsembleAccuracy(valid_x, valid_y)) + '\n\n')
f.write('Possible uncertainties: ' + str(ensemble.getPossibleUncertainties()) + '\n')
f.write('Actual uncertainty:     ' + str(erg) + '\n')
f.close()
ensemble.getNetworks()[accuracies.index(max(accuracies))].save_weights(WEIGHTS_FILE)


if args.log_dir:
    sys.stdout.close()
