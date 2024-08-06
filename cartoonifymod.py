import cv2
from source.cartoonize import Cartoonizer
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import logging
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
        self.cartoonStyle = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon_compound-models')
        self.artstyle = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon-artstyle_compound-models')
        self.handdrawn = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon-handdrawn_compound-models')
        self.sketch = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon-sketch_compound-models')
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
            logging.info(f'imageResize initial: {img.size}')
            img = img.convert("RGB")
            img = np.array(img)
            logging.info(f'imageResize: {img.shape}')
        return img[...,::-1]
    def cartoonify(self, img):
        img=self.imageResize(img)
        image=self.cartoonStyle.cartoonize(img)
        return Image.fromarray(image)
    def artstyleFunc(self, img):
        logging.info("working in art style")
        img=self.imageResize(img)
        logging.info(f'{img.shape}')
        return Image.fromarray(self.artstyle.cartoonize(img))
    def handdrawnFunc(self, img):
        img=self.imageResize(img)
        return Image.fromarray (self.handdrawn.cartoonize(img))
    def sketchFunc(self, img):
        img=self.imageResize(img)
        return Image.fromarray(self.sketch.cartoonize(img))
        