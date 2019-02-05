import keras
#keras.__version__

from keras.models import load_model
from keras import backend as K

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


while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, img_original = video_stream.read()

    frame_down = cv2.resize(img_original, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    # increase contrast
    img_yuv = cv2.cvtColor(frame_down, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])# equalize the histogram of the Y channel
    frame_down = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)# convert the YUV image back to RGB format

    # and then remove noise
    frame_down = cv2.fastNlMeansDenoisingColored(frame_down,None,7,7,3,11)

    img = frame_down.copy() # make a copy for numpy

    img = image.img_to_array(img[:,:,::-1]) # convert to numpy


    img_original = cv2.resize(img_original, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    x = np.expand_dims(img, axis=0)

    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    x = preprocess_input(x)

    preds = model.predict(x)
    text = str(decode_predictions(preds, top=1)[0])
    print('Predicted:', text)

    predicted_class = np.argmax(preds)

    predicted_class_output = model.output[:, predicted_class]

    # The is the output feature map of the `block5_conv3` layer,
    # the last convolutional layer in VGG16
    last_conv_layer = model.get_layer('block5_conv3')

    # This is the gradient of the predicted class with regard to
    # the output feature map of `block5_conv3`
    grads = K.gradients(predicted_class_output, last_conv_layer.output)[0]

    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the predicted class
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    # We then normalize the heatmap 0-1 for visualization:
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = img_original

    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)

    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img * 0.6
    
    cv2.putText(superimposed_img, text, 
                (5, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), lineType=cv2.LINE_AA)

    cv2.imshow('frame', superimposed_img.astype(np.uint8))
