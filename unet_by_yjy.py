import tensorflow as tf
import numpy as np

def conv2d(x, w_shape, b_shape, keep_prob_):
    weights = tf.get_variable("conv_weights", w_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    conv_2d = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable("biases", b_shape,
                             initializer=tf.constant_initializer(0.1))
    droped = tf.nn.dropout(conv_2d, keep_prob_)
    return tf.nn.relu(tf.nn.bias_add(droped, biases))


def conv(x, w_shape, b_shape):
    weights = tf.get_variable("weights", w_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", b_shape,
                             initializer=tf.constant_initializer(0.1))
    return tf.nn.relu(tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='VALID') + biases)


def deconv2d(x, w_shape, b_shape, stride):
    x_shape = tf.shape(x)
    weights = tf.get_variable("deconv_weights", w_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", b_shape,
                             initializer=tf.constant_initializer(0.1))

    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1, stride, stride, 1],
                                  padding='VALID') + biases


def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')


def gen_model(x, keep_prob):
    channel=21
    with tf.variable_scope("conv11"):
        conv11 = conv2d(x, [3, 3, 1, 32], [32], keep_prob)
    with tf.variable_scope("conv12"):
        conv12 = conv2d(conv11, [3, 3, 32, 32], [32], keep_prob)
    with tf.variable_scope("maxpool1"):
        pool1 = max_pool(conv12,2)

    with tf.variable_scope("conv21"):
        conv21 = conv2d(pool1, [3, 3, 32, 64], [64], keep_prob)
    with tf.variable_scope("conv22"):
        conv22 = conv2d(conv21, [3, 3, 64, 64], [64], keep_prob)
    with tf.variable_scope("maxpool2"):
        pool2 = max_pool(conv22,2)

    with tf.variable_scope("conv31"):
        conv31 = conv2d(pool2, [3, 3, 64, 128], [128], keep_prob)
    with tf.variable_scope("conv32"):
        conv32 = conv2d(conv31, [3, 3, 128, 128], [128], keep_prob)
    with tf.variable_scope("maxpool3"):
        pool3 = max_pool(conv32,2)

    with tf.variable_scope("conv41"):
        conv41 = conv2d(pool3, [3, 3, 128, 256], [256], keep_prob)
    with tf.variable_scope("conv42"):
        conv42 = conv2d(conv41, [3, 3, 256, 256], [256], keep_prob)
    with tf.variable_scope("maxpool4"):
        pool4 = max_pool(conv42,2)


    with tf.variable_scope("l_conv1"):
        l_conv1 = conv2d(pool4, [3, 3, 256, 512], [512], keep_prob)
    with tf.variable_scope("l_conv2"):
        l_conv2 = conv2d(l_conv1, [3, 3, 512, 512], [512], keep_prob)

    with tf.variable_scope("deconv4"):
        deconv4 = deconv2d(l_conv2, [2, 2, 256, 512], [256], 2)
        deconv_concat4 = tf.nn.relu(tf.concat([conv42, deconv4], axis = 3))
    with tf.variable_scope("conv51"):
        conv51 = conv2d(deconv_concat4, [3, 3, 512, 256], [256], keep_prob)
    with tf.variable_scope("conv52"):
        conv52 = conv2d(conv51, [3, 3, 256, 256], [256], keep_prob)

    with tf.variable_scope("deconv3"):
        deconv3 = deconv2d(conv52, [2, 2, 128, 256], [128], 2)
        deconv_concat3 = tf.nn.relu(tf.concat([conv32, deconv3], axis = 3))
    with tf.variable_scope("conv61"):
        conv61 = conv2d(deconv_concat3, [3, 3, 256, 128], [128], keep_prob)
    with tf.variable_scope("conv62"):
        conv62 = conv2d(conv61, [3, 3, 128, 128], [128], keep_prob)

    with tf.variable_scope("deconv2"):
        deconv2 = deconv2d(conv62, [2, 2, 64, 128], [64], 2)
        deconv_concat2 = tf.nn.relu(tf.concat([conv22, deconv2], axis=3))
    with tf.variable_scope("conv71"):
        conv71 = conv2d(deconv_concat2, [3, 3, 128, 64], [64], keep_prob)
    with tf.variable_scope("conv72"):
        conv72 = conv2d(conv71, [3, 3, 64, 64], [64], keep_prob)

    with tf.variable_scope("deconv1"):
        deconv1 = deconv2d(conv72, [2, 2, 32, 64], [32], 2)
        deconv_concat1 = tf.nn.relu(tf.concat([conv12, deconv1], axis=3))
    with tf.variable_scope("conv81"):
        conv81 = conv2d(deconv_concat1, [3, 3, 64, 32], [32], keep_prob)
    with tf.variable_scope("conv82"):
        conv82 = conv2d(conv81, [3, 3, 32, 32], [32], keep_prob)

    with tf.variable_scope("out") as scope:
        out_image = conv(conv82,[1,1,32,channel],[channel]) #output = [10,512,512,9] #segment_data = [10,512,512]
        return out_image

def loss(logits, labels):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)








def read_image(img_list , batch_size):
    image = []
    for i in range(0,batch_size):
        image.append(np.load(img_list[i]))
    image = np.array(image)
    del img_list[0:batch_size]
    return image


