import tensorflow as tf
import tflearn
from tflearn.data_utils import image_preloader
import numpy as np
import os

#####directories
TRAIN_DATA='/home/kpranav1998/PycharmProjects/num_fingers/train_data.txt'
MY_DATA='/home/kpranav1998/PycharmProjects/num_fingers/test.txt'
TEST_DATA='/home/kpranav1998/PycharmProjects/num_fingers/test_data.txt'
img_size=128
######image loading
num_classes=6
x_train,y_train=image_preloader(TRAIN_DATA,mode='file',image_shape=(img_size,img_size),categorical_labels=True,normalize=True,grayscale=True)
x_test,y_test=image_preloader(TEST_DATA,mode='file',image_shape=(img_size,img_size),categorical_labels=True,normalize=True,grayscale=True)
x_my_test,y_my_test=image_preloader(MY_DATA,mode='file',image_shape=(img_size,img_size),categorical_labels=True,normalize=True,grayscale=True)

x_train = np.reshape(x_train, [len(x_train), img_size, img_size, 1])
x_test = np.reshape(x_test, [len(x_test), img_size, img_size, 1])
x_my_test = np.reshape(x_my_test, [len(x_my_test), img_size, img_size, 1])



###placeholders
x=tf.placeholder(name="x",shape=[None,img_size,img_size,1],dtype=tf.float32)
y=tf.placeholder(name="y",shape=[None,num_classes],dtype=tf.float32)
print(y_train[0].shape)
#######model


input_layer=x
#filters
filter1= tf.get_variable("filter1",shape=[5,5,1,img_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
filter2 = tf.get_variable("filter2",shape=[3,3,img_size,32],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())

#layer 1
conv1=tf.nn.conv2d(input=x,filter=filter1,strides=[1,1,1,1],padding='SAME',name="conv1")
relu1=tf.nn.relu(conv1,name='relu1')
pool1=out_layer1=tflearn.layers.conv.max_pool_2d(relu1, kernel_size=5)
conv2=tf.nn.conv2d(input=pool1,filter=filter2,strides=[1,1,1,1],padding='SAME',name="conv2")
relu2=tf.nn.relu(conv2,name='relu2')
pool2=tflearn.layers.conv.max_pool_2d(relu2, kernel_size=3)
flat_array=tflearn.layers.core.flatten(pool2,name="flat_array")
length=int(flat_array.shape[1])
print(length)

#initialisers
w1_init= tf.cast(np.random.rand(length,img_size*img_size)*np.sqrt(2/length), tf.float32)

w2_init=tf.cast(np.random.rand(img_size*img_size,6)*np.sqrt(2/img_size*img_size),tf.float32)
w1=tf.get_variable(dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(uniform=True,
    seed=None,dtype=tf.float32),name="W1",shape=(length,img_size*img_size))
w2=tf.get_variable(dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(uniform=True,
    seed=None,dtype=tf.float32),shape=(img_size*img_size,6),name="W2")

l1=tf.matmul(tf.transpose(w1),tf.transpose(flat_array))
l2=tf.nn.relu(l1)


l3=tf.matmul(tf.transpose(w2),l2)
y_predicted=tf.nn.softmax(logits=tf.transpose(l3),name='y_predicted')
#cost
cost=tf.reduce_mean(-tf.multiply(y,tf.log(y_predicted))+tf.multiply((y-1),tf.log(1-y_predicted)))
print("5")

#optimizer
optimizer=tf.train.AdamOptimizer(learning_rate=0.0000001,beta1=0.9,beta2=0.99).minimize(cost)
init=tf.global_variables_initializer()

#accuracy
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_predicted,1)),dtype=tf.float32))

epochs = 2
batch_size=64
no_itr_per_epoch=len(x_train)//batch_size
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(init)
    for iteration in range(epochs):
        print("Iteration no: {} ".format(iteration))

        previous_batch = 0
        # Do our mini batches:
        for i in range(no_itr_per_epoch):
            current_batch = previous_batch + batch_size
            x_input = x_train[previous_batch:current_batch]
            x_images = np.reshape(x_input, [batch_size, img_size, img_size,1])

            y_input = y_train[previous_batch:current_batch]
            y_label = np.reshape(y_input, [batch_size, num_classes])
            previous_batch = previous_batch + batch_size

            _, loss= sess.run([optimizer, cost], feed_dict={x: x_images, y: y_label})
            print(loss)


    Accuracy_val = sess.run(accuracy,
                                feed_dict={
                                    x: x_train,
                                    y: y_train
                                })
    Accuracy_val = round(Accuracy_val * 100, 2)

    print("trainig accuracy"+str(Accuracy_val))
    Accuracy_val2 = sess.run(accuracy,
                            feed_dict={
                                x: x_test,
                                y: y_test
                            })
    Accuracy_val2 = round(Accuracy_val2 * 100, 2)

    print("test_accuracy"+str(Accuracy_val2))
    save_path = saver.save(sess, "/home/kpranav1998/PycharmProjects/num_fingers/model2.ckpt")
    print("Model saved in file: %s" % save_path)
    '''model = tflearn.DNN(y_predicted, checkpoint_path='model2.tfl.ckpt')
    model.save("model.tfl")'''


    ans=sess.run(y_predicted,feed_dict={x:x_my_test,y:y_my_test})

    print("anwser is \n")
    print(ans)
    ans2=np.argmax(ans,1)
    answer=np.reshape(ans2,newshape=[len(x_my_test),1])
    y_test_temp=np.argmax(y_my_test,1)
    y_test_temp = np.reshape(y_test_temp, newshape=[len(x_my_test), 1])
    answer=np.hstack((answer,y_test_temp))

    print(answer)

