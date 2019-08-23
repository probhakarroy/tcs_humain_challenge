#env/bin/python

import tensorflow as tf
from model_utils import dataset_generator
import os, argparse

#argument parsers
parser = argparse.ArgumentParser(description='Multi-Task Learning Network.')
parser.add_argument('--evaluate', help='Evaluate the model.',
                    dest='evaluate', action='store_true')
parser.add_argument('--train', help='Train the model.',
                    dest='evaluate', action='store_false')
parser.set_defaults(evaluate=True)
args = parser.parse_args()


train_batches, validation_batches, test_batches = dataset_generator.create()


#multi-task learning model
inputs = tf.keras.layers.Input(shape = (299, 299, 3))

#feature extractor
convnet = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
convnet = tf.keras.layers.MaxPooling2D((5, 5), padding='same')(convnet)

convnet = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(convnet)
convnet = tf.keras.layers.MaxPooling2D((5, 5), padding='same')(convnet)

convnet = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='same')(convnet)
convnet = tf.keras.layers.MaxPooling2D((5, 5), padding='same')(convnet)

convnet = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(convnet)
convnet = tf.keras.layers.MaxPooling2D((5, 5), padding='same')(convnet)

convnet = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(convnet)
x = tf.keras.layers.GlobalAveragePooling2D()(convnet)

x = tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)

#emotion prediction head
emotion_pred = tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())(x)
emotion_pred = tf.keras.layers.Dropout(0.2)(emotion_pred)
emotion_pred = tf.keras.layers.Dense(5, activation = 'softmax', name = 'emotions')(emotion_pred)

#age prediction head
age_pred = tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())(x)
age_pred = tf.keras.layers.Dropout(0.2)(age_pred)
age_pred = tf.keras.layers.Dense(6, activation = 'softmax', name = 'age')(age_pred)


#ethinicity prediction head
ethinicity_pred = tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())(x)
ethinicity_pred = tf.keras.layers.Dropout(0.2)(ethinicity_pred)
ethinicity_pred = tf.keras.layers.Dense(6, activation = 'softmax', name = 'ethinicity')(ethinicity_pred)


model = tf.keras.models.Model(inputs = inputs , outputs = [emotion_pred, age_pred, ethinicity_pred])

#callbacks
try :
    os.mkdir('./tcs_fr_weights')
except :
    print('tcs_fr_weights directory already exist!')

log = tf.keras.callbacks.CSVLogger('./tcs_fr_log.csv')
checkpoint = tf.keras.callbacks.ModelCheckpoint('./tcs_fr_weights/weights-{epoch:02d}.h5', monitor = 'val_loss',
                                          save_best_only = True, save_weights_only = True, verbose = 1)
lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose = 1, factor = 0.5, patience = 3, min_lr = 0.00005)



#compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

if args.evaluate :
    #evaluate model with best learned weights
    model.load_weights('./model_data/weights/weights-50.h5')
    model.evaluate(test_batches)
else :
    #training the model
    history = model.fit(train_batches, epochs = 50, validation_data = validation_batches, callbacks = [log, checkpoint, lr_decay])
    #saving the model
    model.save_weights('./weight_end_epoch')
