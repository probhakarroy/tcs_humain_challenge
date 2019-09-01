# Facial Recognition
<p>
  <img src="https://img.shields.io/badge/version-0.0.1-blue.svg?cacheSeconds=2592000" />
  <a href="https://github.com/probhakarroy/tcs_humain_challenge#readme">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" target="_blank" />
  </a>
  <a href="https://github.com/probhakarroy/tcs_humain_challenge/graphs/commit-activity">
    <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" target="_blank" />
  </a>
  <a href="https://github.com/probhakarroy/tcs_humain_challenge/blob/master/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" target="_blank" />
  </a>
</p>

> Deep Multi-Task Learning Network For Emotion, Age, Ethinicity Classification.<br>
> Created For TCS HumAIn Challenge.

### 🏠 [Homepage](https://probhakarroy.github.io/tcs_humain_challenge/)

## Install Dependancy
```
$pip3 install --upgrade -r requirements.txt
```

## Usage
```
$python3 model.py --help
usage: model.py [-h] [--predict] [--evaluate] [--train] [--epoch EPOCH]
                [--epoch_weight EPOCH_WEIGHT]

Multi-Task Learning Network.

optional arguments:
  -h, --help            show this help message and exit
  --predict             Use the model to predict an image from url.
  --evaluate            Evaluate the model.
  --train               Train the model.
  --epoch EPOCH         No. of Epochs. [Default : 50]
  --epoch_weight EPOCH_WEIGHT
                        Load the weight from trained epoch. [Choices : 23, 26,
                        49, 50] [Default : 50]
```

A Deep Multi-Task Learning Network For Emotion, Age, Ethinicity Classification trained 
using the dataset provided by TCS of 120 labeled images [Face_Recognition.json] using TensorFlow.

## Dataset
The dataset was provided by TCS HumAIn in a JSON File.<br>
The dataset generator script [dataset_generator.py] in model_utils is used to parse the 
json file and download the images then reshape and covert them to tf.Tensor of shape ```299 x 299 x 3``` and also parse the labels and encode them to one-hot vectors.<br>

Some Data Samples :- <br>
![](model_data/samples/1.jpeg)
![](model_data/samples/2.jpeg)
<br>

Two datapoints out of 120 datapoints cannot be decoded using TensorFlow hence 118 images 
were processed with tensorflow and then data augmented to produce 472 datapoints and then
used to create three tf.data.Dataset input pipeline for training, validation and testing
of the tensorflow model.

```
  Total no. of datapoints after data augmentation : 472
	Total no. of datapoints in train set : 354
	Total no. of datapoints in validation set : 94
	Total no. of datapoints in test set : 23
```

Classes :-<br>
```sh
Emotion : {'Emotion_Neutral', 'Not_Face', 'Emotion_Sad', 'Emotion_Angry', 'Emotion_Happy'}
Age : {'Age_above_50', 'Age_30_40', 'Age_20_30', 'Age_40_50', 'Age_below20', 'others'}
Ethinicity : {'E_Hispanic', 'E_White', 'E_Black', 'E_Asian', 'E_Indian', 'others'}
```


## Model Architecture
![](model_data/model.png)

## Training
The model was trained on Google Colab using Nvidia Tesla T4 GPU.

Usage Script :-<br>
```sh
$python3 model.py --train
#OR
$python3 model.py --train --epoch 50
```

Some Epoch Metrics :-<br>

```
Epoch 1/50
21/22 [===========================>..] - ETA: 1s - loss: 6.2680 - emotions_loss: 1.4739 - age_loss: 1.7252 - ethinicity_loss: 1.5602 - emotions_accuracy: 0.3333 - age_accuracy: 0.2470 - ethinicity_accuracy: 0.3929
Epoch 00001: saving model to ./tcs_fr_weights/weights-01.h5
22/22 [==============================] - 39s 2s/step - loss: 6.2826 - emotions_loss: 1.4719 - age_loss: 1.7199 - ethinicity_loss: 1.5878 - emotions_accuracy: 0.3438 - age_accuracy: 0.2528 - ethinicity_accuracy: 0.3835 - val_loss: 5.8135 - val_emotions_loss: 1.4673 - val_age_loss: 1.5578 - val_ethinicity_loss: 1.4182 - val_emotions_accuracy: 0.1875 - val_age_accuracy: 0.4375 - val_ethinicity_accuracy: 0.5000

...

Epoch 23/50
21/22 [===========================>..] - ETA: 1s - loss: 1.1570 - emotions_loss: 0.1428 - age_loss: 0.2857 - ethinicity_loss: 0.1560 - emotions_accuracy: 0.9554 - age_accuracy: 0.9167 - ethinicity_accuracy: 0.9673
Epoch 00023: saving model to ./tcs_fr_weights/weights-23.h5
22/22 [==============================] - 38s 2s/step - loss: 1.1737 - emotions_loss: 0.1398 - age_loss: 0.2891 - ethinicity_loss: 0.1723 - emotions_accuracy: 0.9574 - age_accuracy: 0.9148 - ethinicity_accuracy: 0.9602 - val_loss: 3.0625 - val_emotions_loss: 0.7331 - val_age_loss: 0.9864 - val_ethinicity_loss: 0.7717 - val_emotions_accuracy: 0.6875 - val_age_accuracy: 0.8125 - val_ethinicity_accuracy: 0.8125

...

Epoch 49/50
21/22 [===========================>..] - ETA: 1s - loss: 0.9638 - emotions_loss: 0.1036 - age_loss: 0.1933 - ethinicity_loss: 0.1210 - emotions_accuracy: 0.9643 - age_accuracy: 0.9583 - ethinicity_accuracy: 0.9881
Epoch 00049: saving model to ./tcs_fr_weights/weights-49.h5
22/22 [==============================] - 39s 2s/step - loss: 0.9751 - emotions_loss: 0.1055 - age_loss: 0.1942 - ethinicity_loss: 0.1297 - emotions_accuracy: 0.9631 - age_accuracy: 0.9574 - ethinicity_accuracy: 0.9830 - val_loss: 3.4741 - val_emotions_loss: 0.8931 - val_age_loss: 1.2171 - val_ethinicity_loss: 0.8187 - val_emotions_accuracy: 0.6875 - val_age_accuracy: 0.6875 - val_ethinicity_accuracy: 0.8125
Epoch 50/50
21/22 [===========================>..] - ETA: 1s - loss: 0.9500 - emotions_loss: 0.1099 - age_loss: 0.1817 - ethinicity_loss: 0.1136 - emotions_accuracy: 0.9643 - age_accuracy: 0.9732 - ethinicity_accuracy: 0.9851
Epoch 00050: saving model to ./tcs_fr_weights/weights-50.h5
22/22 [==============================] - 39s 2s/step - loss: 0.9638 - emotions_loss: 0.1094 - age_loss: 0.1861 - ethinicity_loss: 0.1237 - emotions_accuracy: 0.9631 - age_accuracy: 0.9688 - ethinicity_accuracy: 0.9801 - val_loss: 3.7255 - val_emotions_loss: 0.9669 - val_age_loss: 1.2879 - val_ethinicity_loss: 0.9266 - val_emotions_accuracy: 0.6875 - val_age_accuracy: 0.6875 - val_ethinicity_accuracy: 0.7500

```

Model Metrics :-<br>
![](model_data/metrics/1.jpg)
![](model_data/metrics/2.jpg)
![](model_data/metrics/3.jpg)
![](model_data/metrics/4.jpg)
![](model_data/metrics/5.jpg)

Saved trained epoch weights of the model with best validation accuracies for the
epoch - 23, 26, 49, 50 can be found in ```model_data/weights``` folder. 


## Validation
Best Validation Accuracy for the model for Epoch 23 :-<br>

```sh
Emotion Accuracy : 68.75%
Age Accuracy : 81.25%
Ethinicity : 81.25%
```

## Evaluate

Usage Script :-<br>
```sh
$python3 model.py --evaluate
#OR
$python3 model.py --evaluate --epoch_weight 50
```

Test Metrics for the model with best epoch weights :-<br>

```
$python3 model.py --evaluate

Loading the links :-
100%|██████████████████████████████████████████████████████████████████████| 120/120 [00:00<00:00, 268722.09it/s]

Downloading the images and converting them to tf tensors:-
100%|██████████████████████████████████████████████████████████████████████| 119/119 [02:11<00:00,  1.11s/it]

Data Augmentations:-
100%|██████████████████████████████████████████████████████████████████████| 118/118 [00:01<00:00, 97.08it/s]


	Total no. of datapoints after data augmentation : 472
	Total no. of datapoints in train set : 354
	Total no. of datapoints in validation set : 94
	Total no. of datapoints in test set : 23

23/23 [==============================] - 2s 71ms/step - loss: 5.9781 - emotions_loss: 1.5415 - age_loss: 1.9576 - ethinicity_loss: 1.9338 - emotions_accuracy: 0.6522 - age_accuracy: 0.5217 - ethinicity_accuracy: 0.5652
```

## Prediction

Usage Script :-<br>
```sh
$python3 model.py --predict
#OR
$python3 model.py --predict --epoch_weight 50
```

Sample Prediction of the model :- <br>
```sh
$python3 model.py --predict
Enter Image Path/Url : http://com.dataturks.a96-i23.open.s3.amazonaws.com/2c9fafb06477f4cb0164895548a600a3/66127d05-93eb-498f-bac3-85a19bcbbbc7___2538464.main_image.jpg.jpeg

Downloading the image and converting it to tf tensors...
Prediction :-
        predicted emotion : Emotion_Happy
        predicted age : Age_below20
        predicted ethinicity : E_White
``` 

Output Screen :- <br>
![](model_data/prediction/pred_1.png)

<br>

```sh
$python3 model.py --predict
Enter Image Path/Url : http://com.dataturks.a96-i23.open.s3.amazonaws.com/2c9fafb06477f4cb0164895548a600a3/e3f39fd4-8888-4eea-a49d-038f70a8c540___instagram-famous-clothing-stores.jpg.jpeg

Downloading/Opening the image and converting it to tf tensors...
Prediction :-
        predicted emotion : Emotion_Happy
        predicted age : Age_20_30
        predicted ethinicity : E_Hispanic
``` 

Output Screen :- <br>
![](model_data/prediction/pred_2.png)


```sh
Tested In Ubuntu-19.04 with Python 3.7.3.
```