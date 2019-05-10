import tensorflow as tf
import numpy as np
from unet_all_label import vgg19
import datafunction as df
import time

#vgg_loss1 = vgg.Vgg19()
#loss1 = vgg_loss1.build(df.tensortoimage(vgg_out_image[:,:,:,0]))

def run_vgg19(in_image,channel,name=None):
    with tf.variable_scope(name):
        start_time = time.time()
        print("build model started")
        out_image1=[]
        out_image2=[]
        for i in range(0,channel):
            with tf.variable_scope("%s%d"%(name,i)):
                temp = vgg19.Vgg19()
                temp_img1, temp_img2 = temp.build(df.tensortoimage(in_image[:,:,:,i]))
                out_image1.append(temp_img1)
                out_image2.append(temp_img2)
        ret1 = tf.stack(out_image1, axis=-1)
        ret2 = tf.stack(out_image2, axis=-1)
        print(("build model finished: %ds" % (time.time() - start_time)))
        return ret1, ret2