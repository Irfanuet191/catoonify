import cv2
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import logging
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
baseDir = os.path.abspath(os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO,  # Set the desired logging level
                    # now format has acstime, level, location , message
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Log to the console
                        logging.FileHandler(os.path.join(baseDir, "cartoon.log"))  # Log to a file
                        # Add more handlers as needed, e.g., logging.FileHandler('app.log')
                    ])

class Cartoonifier:
    def __init__(self):
       
        self.cartoonStyle = pipeline(Tasks.image_portrait_stylization,
                       model='damo/cv_unet_person-image-cartoon-artstyle_compound-models')     
        self.sketch = pipeline(Tasks.image_portrait_stylization,
                       model='damo/cv_unet_person-image-cartoon-sketch_compound-models')
        self.catoon3d=pipeline(Tasks.image_portrait_stylization,"damo/cv_unet_person-image-cartoon-3d_compound-models")
       
        
    def imageResize(self,img):
        logging.info(f'imageResize working: {img.size}')
        width, height = img.size
        if width*height >= 1024*1024:
            img = img.convert("RGB")
            width, height = img.size
            while width*height >= 1024*1024:
                width = int(width*0.8)
                height = int(height*0.8)
            img = img.resize((width, height))
            img = np.array(img)
        else:
            img = img.convert("RGB")
            img = np.array(img)
        return img
    def cartoonify(self, img):
        img=self.imageResize(img)
        ImageFile=self.cartoonStyle(input=img)
        image=Image.fromarray(ImageFile['output_img'].astype(np.uint8))
        return image
    def sketchFunc(self, img):
        img=self.imageResize(img)
        ImageFile=self.sketch(input=img)
        image=Image.fromarray(ImageFile['output_img'].astype(np.uint8))
        return image
    def cartoon3dFunc(self, img):
        img=self.imageResize(img)
        ImageFile=self.catoon3d(input=img)
        image=Image.fromarray(ImageFile['output_img'].astype(np.uint8))
        return image