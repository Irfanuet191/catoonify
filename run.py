import cv2
from source.cartoonize import Cartoonizer
import os
import tensorflow as tf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
physical_devices = tf.config.experimental.list_physical_devices()
from PIL import Image
import numpy as np

def process():

    # algo = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon_compound-models')
    # algo2 = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon-artstyle_compound-models')
    # # algo3 = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon-artstyle_compound-models')
    # algo4 = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon-handdrawn_compound-models')
    # algo5 = Cartoonizer(dataroot='./damo1/cv_unet_person-image-cartoon-sketch_compound-models')



    # img = cv2.imread('/home/irfan/catoonify/DSC_0035.JPG')[...,::-1]
 
    img_cartoon = pipeline(Tasks.image_portrait_stylization,
                       model='damo/cv_unet_person-image-cartoon-artstyle_compound-models')
    # img = cv2.resize(img, (256, 256))
    imagefile=cv2.imread('/home/irfan/catoonify/DSC_0035.JPG')   
    
    image=img_cartoon(input=imagefile)

    # result = img_cartoon.cartoonize(img)
    cv2.imwrite("ouput.png",image["output_img"])
    print(image["output_img"].shape)
    random_array = image["output_img"].astype(np.uint8)
    random_image = Image.fromarray(random_array)
    # image=Image.fromarray(np.array(image["output_img"]))

    # cv2.imwrite('res4.png', image)
    print('finished!')
    print("Physical devices:", physical_devices)



if __name__ == '__main__':
    process()



