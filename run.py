import cv2
from source.cartoonize import Cartoonizer
import os
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices()

def process():

    algo = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon_compound-models')
    algo2 = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon-artstyle_compound-models')
    # algo3 = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon-artstyle_compound-models')
    algo4 = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon-handdrawn_compound-models')
    algo5 = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon-sketch_compound-models')



    img = cv2.imread('/home/irfan/catoonify/RGB.PNG')[...,::-1]

    result = algo5.cartoonize(img)

    cv2.imwrite('res4.png', result)
    print('finished!')
    print("Physical devices:", physical_devices)



if __name__ == '__main__':
    process()



