import glob
import numpy as np
import tensorflow as tf
from unet_all_label import unet_by_yjy as uf
import gc
import datafunction as df
from unet_all_label import vgg19 as vgg
from unet_all_label.run_vgg19 import run_vgg19


epoch1=0
epoch2=20
batch_size = 10
learning_rate = 0.0006
image_width = 128
merge_size = 5

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

keep_prob = tf.placeholder(tf.float32)
in_image = tf.placeholder(tf.float32,[batch_size,image_width,image_width,1])
seg_image = tf.placeholder(tf.int32,[batch_size,image_width,image_width])
out_image = uf.gen_model(in_image,keep_prob)
l2_loss =tf.reduce_mean(uf.loss(out_image,seg_image))*7
#optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
vgg_seg_image = in_image*tf.one_hot(seg_image,depth=8, axis=-1)
vgg_out_image = in_image*out_image
print(in_image, tf.one_hot(seg_image,depth=8, axis=-1), out_image)
print(vgg_seg_image,vgg_out_image)
loss1_1,loss1_2=run_vgg19(vgg_out_image,channel=8,name="result_image")
loss2_1,loss2_2=run_vgg19(vgg_seg_image,channel=8,name="segment_image")
print(loss1_1,loss1_2)
print(loss2_1,loss2_2)
vgg_loss=tf.norm((loss1_1-loss2_1))/400000
vgg_loss2= tf.norm((loss1_2-loss2_2))/2000000
print(vgg_loss)
print(vgg_loss2)
loss= l2_loss+vgg_loss+vgg_loss2
optimizer_ = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)



sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=100)
saver.restore(sess, './0.9scale/0.9scale_l=1-12')


##df.npdata_merge('file_name', 'input_directory', 'output_directory', merge_size)##

##df.npdata_merge('seg_merge', 'D:\_unet\seg_data\*.npy', 'D:\_unet\seg_merge', 13000)##
#df.npdata_merge('qsm_merge', 'D:\_unet\qsm_data\*.npy', 'D:\_unet\qsm_merge', 13000)
#df.npdata_merge('seg_noseg_merge','D:\_unet\seg_noseg_data\*.npy','D:\_unet\seg_merge', 13000)
#df.npdata_merge('qsm_noseg_merge','D:\_unet\qsm_noseg_data\*.npy','D:\_unet\qsm_merge', 13000)

seg_list = glob.glob('D:\_unet\seg_merge\*.npy')
qsm_list = glob.glob('D:\_unet\qsm_merge\*.npy')
seg_list, qsm_list = df.shuffle(seg_list,qsm_list,0)

for epochs in range(epoch1,epoch2):


    for files in range(0,seg_list.__len__()//merge_size):
        qsm_merge = df.load_merged_npy_file(qsm_list[files*merge_size:(files+1)*(merge_size)])
        seg_merge = df.load_merged_npy_file(seg_list[files*merge_size:(files+1)*(merge_size)])
        data_size = qsm_merge.shape[0]
        df.shuffle(seg_merge,qsm_merge,0)
        seg_merge = np.reshape(seg_merge, [data_size//batch_size, batch_size, image_width, image_width])
        qsm_merge = np.reshape(qsm_merge, [data_size//batch_size, batch_size, image_width, image_width ,1])
        cost_ =0
        for i in range(0,data_size//batch_size):
            qsm_data = qsm_merge[i]
            seg_data = seg_merge[i]
            cost_, l2_cost,vgg_cost, vgg_cost2, _, o_img = sess.run([loss, l2_loss, vgg_loss, vgg_loss2, optimizer_, out_image], feed_dict = {in_image : qsm_data, seg_image : seg_data, keep_prob: 0.7})
            print('l2_cost=%f,' %l2_cost, 'vgg_44loss=%f' %vgg_cost, 'vgg_34loss=%f'%vgg_cost2)
            o_img = np.argmax(o_img, axis=3)
            if(i%100)==99:
                print('epoch = %d, fileset_number=%d, iter=%d'%(epochs,files, i+1))
                print('cost=%f'%(cost_/100))
                print('mean=%f'%np.mean(o_img))
                cost_=0
    gc.collect()
    saver.save(sess, './0.9scale_vgg_double', global_step=(epochs+1))