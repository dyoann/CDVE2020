import cv2
from tensorflow import keras
import numpy as np

image_width = 224
image_height= 224

def generate_cam(model,base_path,output_path,img_name,target_class):
  img = cv2.imread(base_path+img_name)
  img = cv2.resize(img, (image_width,image_height), interpolation = cv2.INTER_LANCZOS4)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img / 255.
  
  class_weights = model.layers[-2].get_weights()[0]
  conv_model = keras.Model(inputs=model.input, outputs=model.get_layer('GAP').input)
  conv_outputs = conv_model.predict(np.array([img]))[0]
  
  cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
  target_class = target_class
  for i, w in enumerate(class_weights[:, target_class]):
    cam += w * conv_outputs[:, :, i]
        
  cam = np.maximum(cam, 0.)
  cam = cam / np.max(cam) # 0 is red in COLORMAP_JET, revert
  cam = np.array(cam * 255, dtype = np.uint8)
  heatmap_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)/255.
  heatmap_cam = cv2.resize(heatmap_cam, (224,224), interpolation = cv2.INTER_LANCZOS4)
  merged = cv2.addWeighted(heatmap_cam, 0.5, img, 0.5, 0)
  cv2.imwrite(output_path+img_name,merged*255)
