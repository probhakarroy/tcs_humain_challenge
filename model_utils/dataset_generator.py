#env/bin/python
import requests
import tqdm, json, os
import tensorflow as tf

def create() :
  fo = open('Face_Recognition.json')
  data = []
  for datapoint in fo:
    data.append(json.loads(datapoint))

  links = []
  labels = []
  print('\nLoading the links :-')
  for i in tqdm.tqdm(range(len(data))):
    links.append(data[i]['content'])
    try:
      labels.append(data[i]['annotation'][0]['label'])
    except:
      links.pop()

  #image downloading path
  images = []
  re_labels = []
  
  try :
    os.mkdir('dataset')
  except :
    print('dataset directory already exist!')

  print('\nDownloading the images and converting them to tf tensors:-')
  for i in tqdm.tqdm(range(len(links))) :
    res = requests.get(links[i])
    res.raise_for_status()
    fo = open(os.path.join('dataset', str(i) + '.jpg'), 'wb')
    for chunks in res.iter_content(1000000) :
      fo.write(chunks)
    fo.close()

    img = tf.io.read_file(os.path.join('dataset', str(i) + '.jpg'))
    try:
      img = tf.image.decode_image(img, channels=3)
    except:
      continue
    img = tf.image.resize(img, [299, 299])
    img /= 255.0
    if len(labels[i]) == 1:
      for j in range(3):
        labels[i].append('others')
    images.append(tf.squeeze(img))
    re_labels.append(labels[i])

  itr = len(images)
  all_images = [*images]
  all_labels = [*re_labels]

  #Data Augmentations
  print('\nData Augmentations :-')
  for i in tqdm.tqdm(range(itr)):
    #random 90 deg, 180 deg, 270 deg, 360deg rotations
    all_images.append(tf.image.rot90(images[i], tf.random.uniform(
        shape=[], minval=0, maxval=4, dtype=tf.int32)))
    all_labels.append(re_labels[i])

    #random flips
    x = tf.image.random_flip_left_right(images[i])
    all_images.append(tf.image.random_flip_up_down(x))
    all_labels.append(re_labels[i])

    #image color transform
    x = tf.image.random_hue(images[i], 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    all_images.append(tf.image.random_contrast(x, 0.7, 1.3))
    all_labels.append(re_labels[i])

  emotions, age, ethinicity = [], [], []

  for label in all_labels:
    #print(label)
    emotions.append(label[0])
    age.append(label[1])
    ethinicity.append(label[2])

  emotions_dict, age_dict, ethinicity_dict = {}, {}, {}
  for i, j in enumerate(set(emotions)):
    emotions_dict[j] = i

  for i, j in enumerate(set(age)):
    age_dict[j] = i

  for i, j in enumerate(set(ethinicity)):
    ethinicity_dict[j] = i

  for i in range(len(emotions)):
    emotions[i] = emotions_dict[emotions[i]]
    age[i] = age_dict[age[i]]
    ethinicity[i] = ethinicity_dict[ethinicity[i]]

  #converting to one hot vectors
  emotions = tf.keras.utils.to_categorical(emotions)
  age = tf.keras.utils.to_categorical(age)
  ethinicity = tf.keras.utils.to_categorical(ethinicity)

  DATASET_SIZE = len(all_images)
  train_size = int(DATASET_SIZE * 0.75)
  val_size = int(DATASET_SIZE * 0.20)
  test_size = int(DATASET_SIZE * 0.05)


  print('\n\n\tTotal no. of datapoints after data augmentation : ' + str(DATASET_SIZE))
  print('\tTotal no. of datapoints in train set : ' + str(train_size))
  print('\tTotal no. of datapoints in validation set : ' + str(val_size))
  print('\tTotal no. of datapoints in test set : ' + str(test_size))

  dataset = tf.data.Dataset.from_tensor_slices(
      (all_images, (emotions, age, ethinicity)))
  dataset = dataset.shuffle(buffer_size=DATASET_SIZE)
  train_dataset = dataset.take(train_size)
  test_dataset = dataset.skip(train_size)
  validation_dataset = test_dataset.skip(val_size)
  test_dataset = test_dataset.take(test_size)

  train_batches = train_dataset.shuffle(train_size).batch(
      16, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  validation_batches = validation_dataset.batch(
      16, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  test_batches = test_dataset.batch(1, drop_remainder=True).prefetch(
      buffer_size=tf.data.experimental.AUTOTUNE)
  
  return train_batches, validation_batches, test_batches
