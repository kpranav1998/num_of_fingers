import tensorflow as tf
import tflearn
from tflearn.data_utils import image_preloader
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import  cv2



num_classes=6
x=tf.placeholder(name="x",shape=[None,64,64,1],dtype=tf.float32)
y=tf.placeholder(name="y",shape=[None,num_classes],dtype=tf.float32)

#layer 1


#initialisers

MY_DATA='/home/kpranav1998/PycharmProjects/num_fingers/test.txt'
TEST_DATA='/home/kpranav1998/PycharmProjects/num_fingers/test_data.txt'
x_my_test,y_my_test=image_preloader(MY_DATA,mode='file',image_shape=(64,64),categorical_labels=True,normalize=True,grayscale=True)
x_my_test = np.reshape(x_my_test, [len(x_my_test), 64, 64, 1])
x_test,y_test=image_preloader(TEST_DATA,mode='file',image_shape=(64,64),categorical_labels=True,normalize=True,grayscale=True)
x_test = np.reshape(x_test, [len(x_test), 64, 64, 1])

init=tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    new_saver = tf.train.import_meta_graph('model.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    w1=sess.run("W1:0")
    w2=sess.run("W2:0")
    filter1=sess.run("filter1:0")
    filter2=sess.run("filter2:0")
    input_layer = x


    # layer 1
    conv1 = tf.nn.conv2d(input=x, filter=filter1, strides=[1, 1, 1, 1], padding='SAME', name="conv1")
    relu1 = tf.nn.relu(conv1, name='relu1')
    pool1 = out_layer1 = tflearn.layers.conv.max_pool_2d(relu1, kernel_size=5)
    conv2 = tf.nn.conv2d(input=pool1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME', name="conv2")
    relu2 = tf.nn.relu(conv2, name='relu2')
    pool2 = tflearn.layers.conv.max_pool_2d(relu2, kernel_size=3)
    flat_array = tflearn.layers.core.flatten(pool2, name="flat_array")
    l1 = tf.matmul(tf.transpose(w1), tf.transpose(flat_array))
    l2 = tf.nn.relu(l1)

    l3 = tf.matmul(tf.transpose(w2), l2)

    y_predicted = tf.nn.softmax(logits=tf.transpose(l3))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_predicted, 1)), dtype=tf.float32))

    ans = sess.run(y_predicted, feed_dict={x: x_my_test, y: y_my_test})
    
    ans2 = np.argmax(ans, 1)
    answer = np.reshape(ans2, newshape=[len(x_my_test), 1])
    y_test_temp = np.argmax(y_my_test, 1)
    y_test_temp = np.reshape(y_test_temp, newshape=[len(x_my_test), 1])
    answer = np.hstack((answer, y_test_temp))
    print("your answer is "+str(answer[0][0]))
