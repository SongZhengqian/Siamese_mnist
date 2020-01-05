#coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = tf.keras.datasets.mnist
#import siamese_mnist_conv
import cv2

np.set_printoptions(suppress=True)
#mnist = input_data.read_data_sets('./MNIST_data',one_hot=True)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 选两张输入图片
fig, ax = plt.subplots(
    nrows=1,
    ncols=2,
    )
ax = ax.flatten()
#input1 = x_train[y_train == 2][1]
input1 = cv2.imread("C:\\Users\\Y\\Desktop\\20190109_IMG_1509.JPG")
ax[0].imshow(input1, cmap='Greys', interpolation='nearest')
#input2 = x_train[y_train == 3][2]
input2 = cv2.imread("C:\\Users\\Y\\Desktop\\20180421_IMG_3498.JPG")
ax[1].imshow(input2, cmap='Greys', interpolation='nearest')
plt.show()
input1 = np.reshape(input1, (-1, 784))
input2 = np.reshape(input2, (-1, 784))



y_s = np.array(1)
zero = np.zeros(64)
y_s = y_s + zero
# 加载模型
sess = tf.Session()
graph_path=os.path.abspath('./model/siamese_net.meta')
model=os.path.abspath('./model/')

server = tf.train.import_meta_graph(graph_path)
server.restore(sess,tf.train.latest_checkpoint(model))
graph = tf.get_default_graph()
# 打印所有op名字
# for op in graph.get_operations():
#     print(op.name)
#填充feed_dict
x1 = graph.get_tensor_by_name('input_x1/Placeholder:0')
x2 = graph.get_tensor_by_name('input_x2/Placeholder:0')
y = graph.get_tensor_by_name('y/Placeholder:0')
feed_dict={x1:input1,x2:input2,y:y_s}

#第一层卷积+池化
conv1 = graph.get_tensor_by_name('siamese/conv1/conv1:0')
pool1 = graph.get_tensor_by_name('siamese/pool1/pool1:0')
conv1_1 = graph.get_tensor_by_name('siamese/conv1_1/conv1:0')
pool1_1 = graph.get_tensor_by_name('siamese/pool1_1/pool1:0')
#第二层卷积+池化
conv2 = graph.get_tensor_by_name('siamese/conv2/conv2:0')
pool2 = graph.get_tensor_by_name('siamese/pool2/pool2:0')
conv2_1 = graph.get_tensor_by_name('siamese/conv2_1/conv2:0')
pool2_1 = graph.get_tensor_by_name('siamese/pool2_1/pool2:0')
# 可视化

#conv1 特征
r1_relu = sess.run(conv1,feed_dict)
#tf.transpose是将r1_relu转置为[20,1,28,28]
r1_tranpose = sess.run(tf.transpose(r1_relu,[3,0,1,2]))
r1_relu_1 = sess.run(conv1_1,feed_dict)
r1_tranpose_1 = sess.run(tf.transpose(r1_relu_1,[3,0,1,2]))
fig,ax = plt.subplots(nrows=1,ncols=20,figsize=(20,1))
for i in range(20):
    ax[i].imshow(r1_tranpose[i][0])
plt.title('Conv1')
fig1,ax1 = plt.subplots(nrows=1,ncols=20,figsize=(20,1))
for i in range(20):
    ax1[i].imshow(r1_tranpose_1[i][0])
plt.title('Conv1_1')
plt.show()
#pool1特征
max_pool1 = sess.run(pool1,feed_dict)
r1_tranpose = sess.run(tf.transpose(max_pool1,[3,0,1,2]))
max_pool1_1 = sess.run(pool1_1,feed_dict)
r1_tranpose_1 = sess.run(tf.transpose(max_pool1_1,[3,0,1,2]))
fig,ax = plt.subplots(nrows=1,ncols=20,figsize=(20,1))
for i in range(20):
    ax[i].imshow(r1_tranpose[i][0])
plt.title('Pool1')
fig1,ax1 = plt.subplots(nrows=1,ncols=20,figsize=(20,1))
for i in range(20):
    ax1[i].imshow(r1_tranpose_1[i][0])
plt.title('Pool1_1')
plt.show()
