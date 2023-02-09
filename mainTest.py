from pyexpat import model
from unittest import result
import cv2
from keras.models import load_model
from PIL import Image
from matplotlib.pyplot import axes
import numpy as np
model=load_model('BrainTumor20Epohs.h5')
image=cv2.imread('D:\\New folder\\Test\\pred\\pred4.jpg')
img=Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)
input_img=np.expand_dims(img, axis=0)
result=model.predict(input_img)
print(result)
print(image)           