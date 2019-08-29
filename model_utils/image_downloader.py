#env/bin/python
import requests, os, sys
import tensorflow as tf

def url(x):
    print('\nDownloading the image and converting it to tf tensors...')
    res = requests.get(x)
    res.raise_for_status()
    try:
        os.mkdir('temp')
    except :
        print('temp directory already exist!')

    fo = open(os.path.join('temp', 'img.jpg'), 'wb')
    for chunks in res.iter_content(1000000):
        fo.write(chunks)
    fo.close()

    img = tf.io.read_file(os.path.join('temp', 'img.jpg'))
    try:
        img = tf.image.decode_image(img, channels=3)
    except:
        print('Unable to convert the image to tf tensors.')
        sys.exit()
        
    img = tf.image.resize(img, [299, 299])
    img /= 255.0
    
    return tf.squeeze(img)