#env/bin/python

import os, argparse, warnings

#filtering tensorflow future warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from model_utils import dataset_generator, image_downloader
import matplotlib.pyplot as plt


#argument parsers
parser = argparse.ArgumentParser(description = 'Multi-Task Learning Network.')
parser.add_argument('--predict', help = 'Use the model to predict an image from url.',
                    dest = 'predict', action = 'store_true')
parser.add_argument('--evaluate', help = 'Evaluate the model.',
                    dest = 'evaluate', action = 'store_true')
parser.add_argument('--train', help = 'Train the model.',
                    dest = 'evaluate', action = 'store_false')
parser.add_argument('--epoch', help = 'No. of Epochs. [Default : 50]', default = 50, type = int)
parser.add_argument('--epoch_weight', help='Load the weight from trained epoch. [Choices : 23, 26, 49, 50] [Default : 50]', default=50, type=int)
parser.set_defaults(predict = True, evaluate = False)
args = parser.parse_args()

if args.predict and not args.evaluate :
    img = input('Enter Image Url : ')
    img = image_downloader.url(img)
    
    #label mapper dicts
    emotion_label = {0: 'Emotion_Neutral', 1: 'Not_Face',
                     2: 'Emotion_Sad', 3: 'Emotion_Angry', 4: 'Emotion_Happy'}
    age_label = {0: 'Age_above_50', 1: 'Age_30_40', 2: 'Age_20_30',
                 3: 'Age_40_50', 4: 'Age_below20', 5: 'others'}
    ethinicity_label = {0: 'E_Hispanic', 1: 'E_White',
                        2: 'E_Black', 3: 'E_Asian', 4: 'E_Indian', 5: 'others'}
else :
    train_batches, validation_batches, test_batches = dataset_generator.create()


#multi-task learning model
inputs = tf.keras.layers.Input(shape = (299, 299, 3))

#feature extractor with 5*5 convolution weights and 5*5 max pooling layers.
convnet = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
convnet = tf.keras.layers.MaxPooling2D((5, 5), padding='same')(convnet)

convnet = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(convnet)
convnet = tf.keras.layers.MaxPooling2D((5, 5), padding='same')(convnet)

convnet = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='same')(convnet)
convnet = tf.keras.layers.MaxPooling2D((5, 5), padding='same')(convnet)

convnet = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(convnet)
convnet = tf.keras.layers.MaxPooling2D((5, 5), padding='same')(convnet)

convnet = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(convnet)

#reducing the complexity for conv and dense connection weights with average pooling.
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


try :
    os.mkdir('./tcs_fr_weights')
except :
    print('tcs_fr_weights directory already exist!')

#callbacks
log = tf.keras.callbacks.CSVLogger('./tcs_fr_log.csv')
checkpoint = tf.keras.callbacks.ModelCheckpoint('./tcs_fr_weights/weights-{epoch:02d}.h5', monitor = 'val_loss',
                                          save_best_only = False, save_weights_only = True, verbose = 1)
lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose = 1, factor = 0.5, patience = 3, min_lr = 0.00005)



#compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

if args.predict and not args.evaluate :
    #model prediction
    model.load_weights('./model_data/weights/weights-{}.h5'.format(args.epoch_weight))
    pred = model.predict(tf.expand_dims(img, 0))

    #prediction
    print('Prediction :-\n\tpredicted emotion : {}\n\tpredicted age : {}\n\tpredicted ethinicity : {}\n'
    .format(emotion_label[tf.argmax(pred[0][0]).numpy()], age_label[tf.argmax(pred[1][0]).numpy()], ethinicity_label[tf.argmax(pred[2][0]).numpy()]))

    #ploting the image
    plt.imshow(img)
    plt.show()
elif args.evaluate and args.predict :
    #evaluate model with best learned weights
    model.load_weights('./model_data/weights/weights-{}.h5'.format(args.epoch_weight))
    model.evaluate(test_batches)
else :
    #training the model
    history = model.fit(train_batches, epochs = args.epoch, validation_data = validation_batches, callbacks = [log, checkpoint, lr_decay])
    #saving the model
    model.save_weights('./weight_end_epoch')
