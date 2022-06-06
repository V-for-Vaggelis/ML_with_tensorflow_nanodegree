# Basic usage:python python predict.py ./test_images/wild_pansy.jpg flower_classifier.h5
# Options:
# Return the top 3 most likely classes:
# $ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
# Use a label_map.json file to map labels to flower names:
# $ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json

import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import argparse

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='A flower classifying application')
parser.add_argument('im_path', metavar='im', help='Path for infering image')
parser.add_argument('model_path', metavar="m", help='Path for the trained model')
parser.add_argument('--top_k','--top_k', help='Number of most possible classes', required=False)
parser.add_argument('--category_names','--category_names', help='label names file', required=False)
args = vars(parser.parse_args())

im_path = args['im_path']
model_path = args['model_path']
if args['top_k']:
    top_k = int(args['top_k'])
else:
    top_k = 1 # unless specified use only one class

if args['category_names']:
    json_file = args['category_names']
else:
    json_file = 'label_map.json'

with open(json_file, 'r') as f: # read labels
    class_names = json.load(f)

loaded = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}) # load model

# the process_image function, necessary transforms
def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (224, 224))/255.0
    return image


# the predict function
def predict(im_path, model, top_k):
    im = Image.open(im_path)
    test_image = np.asarray(im) # convert objet to np array
    test_image_proc = process_image(test_image)
    prediction = model.predict(np.expand_dims(test_image_proc, axis=0))
    top_probs, top_indices = tf.math.top_k(prediction, top_k) # keep only k best
    print("Top propabilities",top_probs.numpy()[0])
    top_classes = [class_names[str(value)] for value in top_indices.numpy()[0]]
    print('Top classes', top_classes)
    return top_probs.numpy()[0], top_classes

# read Image to infer and plot along with top_k classes
im = Image.open(im_path)
test_image = np.asarray(im)
test_image_proc = process_image(test_image)
probs, classes = predict(im_path, loaded, top_k)
fig, (ax1, ax2) = plt.subplots(figsize=(12,4), ncols=2)
ax1.imshow(test_image_proc)
ax2 = plt.barh(classes[::-1], probs[::-1])
plt.tight_layout()
plt.show()
