import tensorflow as tf
import tflearn
from tflearn.data_utils import image_preloader
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import  cv2



num_classes=6

#layer 1


#initialisers

MY_DATA='/home/kpranav1998/PycharmProjects/num_fingers/test.txt'
TEST_DATA='/home/kpranav1998/PycharmProjects/num_fingers/test_data.txt'
x_my_test,y_my_test=image_preloader(MY_DATA,mode='file',image_shape=(64,64),categorical_labels=True,normalize=True,grayscale=True)
x_my_test = np.reshape(x_my_test, [len(x_my_test), 64, 64, 1])

init=tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    new_saver = tf.train.import_meta_graph('model.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    y_predicted = graph.get_tensor_by_name("y_predicted:0")
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")

    ans = sess.run(y_predicted, feed_dict={x: x_my_test, y: y_my_test})
    ans2 = np.argmax(ans, 1)

    answer = np.reshape(ans2, newshape=[len(x_my_test), 1])
    y_test_temp = np.argmax(y_my_test, 1)
    y_test_temp = np.reshape(y_test_temp, newshape=[len(x_my_test), 1])
    answer = np.hstack((answer, y_test_temp))
    print(answer)
