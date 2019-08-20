#env/bin/python

import tensorflow as tf
from model_utils import dataset_generator
import os

train_batches, validation_batches, test_batches = dataset_generator.create()


#feature extractor
base_model = tf.keras.applications.InceptionV3(include_top = False, weights = 'imagenet')
base_model.trainable = False

#multi-task learning model
inputs = tf.keras.layers.Input(shape = (299, 299, 3))
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)

#emotion prediction head
emotion_pred = tf.keras.layers.Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())(x)
emotion_pred = tf.keras.layers.Dropout(0.2)(emotion_pred)
emotion_pred = tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())(emotion_pred)
emotion_pred = tf.keras.layers.Dropout(0.2)(emotion_pred)
emotion_pred = tf.keras.layers.Dense(5, activation = 'softmax', name = 'emotions')(emotion_pred)

#age prediction head
age_pred = tf.keras.layers.Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())(x)
age_pred = tf.keras.layers.Dropout(0.2)(age_pred)
age_pred = tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())(age_pred)
age_pred = tf.keras.layers.Dropout(0.2)(age_pred)
age_pred = tf.keras.layers.Dense(6, activation = 'softmax', name = 'age')(age_pred)


#ethinicity prediction head
ethinicity_pred = tf.keras.layers.Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())(x)
ethinicity_pred = tf.keras.layers.Dropout(0.2)(ethinicity_pred)
ethinicity_pred = tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())(ethinicity_pred)
ethinicity_pred = tf.keras.layers.Dropout(0.2)(ethinicity_pred)
ethinicity_pred = tf.keras.layers.Dense(6, activation = 'softmax', name = 'ethinicity')(ethinicity_pred)


model = tf.keras.models.Model(inputs = inputs , outputs = [emotion_pred, age_pred, ethinicity_pred])

#callbacks
os.mkdir('./tcs_fr_weights')

log = tf.keras.callbacks.CSVLogger('./tcs_fr_log.csv')
checkpoint = tf.keras.callbacks.ModelCheckpoint('./tcs_fr_weights/weights-{epoch:02d}.h5', monitor = 'val_loss',
                                          save_best_only = True, save_weights_only = True, verbose = 1)
lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose = 1, factor = 0.5, patience = 3, min_lr = 0.00005)



#compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

#training the model
history = model.fit(train_batches, epochs = 50, validation_data = validation_batches, callbacks = [log, checkpoint, lr_decay])

#saving the model
model.save('./face_recognition_model')
