import keras
#keras.__version__

from keras.models import load_model
from keras import backend as K

#from resizeimage import resizeimage

from keras.preprocessing import image
import numpy as np
import requests
from io import BytesIO
from PIL import Image

from keras import models

import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import cv2

import copy

video_stream = cv2.VideoCapture(0)
video_stream.set(3,224)
video_stream.set(4,224)

model = VGG16(weights='imagenet')

LAYER_TO_VISUALIZE = 2


while(True):

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        exit()
    if key == ord('w'):
        LAYER_TO_VISUALIZE -= 1
        cv2.destroyAllWindows()
    if key == ord('e'):
        LAYER_TO_VISUALIZE += 1
        cv2.destroyAllWindows()
    if LAYER_TO_VISUALIZE == 15:
        LAYER_TO_VISUALIZE = 2
    if LAYER_TO_VISUALIZE == 1:
        LAYER_TO_VISUALIZE = 14

    # Extracts the outputs of the top 8 layers:
    layer_outputs = [layer.output for layer in model.layers[LAYER_TO_VISUALIZE-1:LAYER_TO_VISUALIZE+1]]
    # Creates a model that will return these outputs, given the model input:
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    DOWNSAMPLE_LEVEL = [0, 4, 4, 4, 2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1][LAYER_TO_VISUALIZE]
    IMAGES_PER_ROW =   [0, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32][LAYER_TO_VISUALIZE]

    ret, img_original = video_stream.read()
    img_original = cv2.resize(img_original, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    img = img_original.astype(np.float64)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    x = np.expand_dims(img, axis=0)

    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    x = preprocess_input(x)


    activations = activation_model.predict(x)
    #first_layer_activation = activations[0]

    # These are the names of the layers, so can have them as part of our plot
    layer_names = []
    for layer in model.layers[LAYER_TO_VISUALIZE-1:LAYER_TO_VISUALIZE+1]:
        layer_names.append(layer.name)

    images_per_row = IMAGES_PER_ROW

    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        #print(layer_name, layer_activation.shape)
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size//DOWNSAMPLE_LEVEL * n_cols, images_per_row * size//DOWNSAMPLE_LEVEL, 3))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                channel_image = channel_image[::DOWNSAMPLE_LEVEL, ::DOWNSAMPLE_LEVEL]
                channel_image = cv2.applyColorMap(channel_image, cv2.COLORMAP_OCEAN)
                display_grid[col * size//DOWNSAMPLE_LEVEL : (col + 1) * size//DOWNSAMPLE_LEVEL,
                             row * size//DOWNSAMPLE_LEVEL : (row + 1) * size//DOWNSAMPLE_LEVEL] = channel_image

        # Display the grid
        cv2.imshow(layer_name, display_grid.astype(np.uint8))
        break

exit()
