# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.reset_default_graph()#重置default_graph计算图以及nodes节点
#当在搭建网络查看计算图时，如果重复运行程序会导致重定义报错。
#为了可以在同一个线程或者交互式环境中重复调试计算图，就需要使用这个函数来重置计算图，随后修改计算图再次运行。

mnist = input_data.read_data_sets('./MNIST_data',one_hot=True)#启用独热编码
print('number of train datasets:',mnist.train.num_examples)
print('number of validation datasets:',mnist.validation.num_examples)
print('number of test datasets:',mnist.test.num_examples)

#定义siamese网络损失函数

def loss_with_step(out1,out2,y):
    margin = 0.5
    labels_t = y
    labels_f = tf.subtract(1.0, y, name="1-yi")          # labels_ = !labels;
    eucd2 = tf.pow(tf.subtract(out1, out2), 2)
    eucd2 = tf.reduce_sum(eucd2, 1)
    eucd = tf.sqrt(eucd2+1e-6, name="eucd")
    C = tf.constant(margin, name="C")
    pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
    neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
    losses = tf.add(pos, neg, name="losses")
    loss = tf.reduce_mean(losses, name="loss")
    return loss
def siamese_loss(out1,out2,y,Q=1):

    Q = tf.constant(Q, name="Q",dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1-out2),1))
    pos = tf.multiply(tf.multiply(y,2/Q),tf.square(E_w))
    neg = tf.multiply(tf.multiply(1-y,2*Q),tf.exp(-2.77/Q*E_w))
    loss = pos + neg
    loss = tf.reduce_mean(loss)
    return loss
#定义siamese网络结构
def siamese(inputs,keep_prob):#inputs=[?,28,28,1]
        with tf.name_scope('conv1') as scope:
            w1 = tf.Variable(tf.truncated_normal(shape=[3,3,1,32],stddev=0.05),name='w1')#W=[heigh,width,input_channel,output_channel]
            b1 = tf.Variable(tf.zeros(32),name='b1')#b=[output_channel]
            conv1 = tf.nn.conv2d(inputs,w1,strides=[1,1,1,1],padding='SAME',name='conv1')
        with tf.name_scope('relu1') as scope:
            relu1 = tf.nn.relu(tf.add(conv1,b1),name='relu1')

        with tf.name_scope('conv2') as scope:
            w2 = tf.Variable(tf.truncated_normal(shape=[3,3,32,64],stddev=0.05),name='w2')
            b2 = tf.Variable(tf.zeros(64),name='b2')
            conv2 = tf.nn.conv2d(relu1,w2,strides=[1,2,2,1],padding='SAME',name='conv2')
        with tf.name_scope('relu2') as scope:
            relu2 = tf.nn.relu(conv2+b2,name='relu2')

        with tf.name_scope('conv3') as scope:
            w3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,128],mean=0,stddev=0.05),name='w3')
            b3 = tf.Variable(tf.zeros(128),name='b3')
            conv3 = tf.nn.conv2d(relu2,w3,strides=[1,2,2,1],padding='SAME',name='conv3')
        with tf.name_scope('relu3') as scope:
            relu3 = tf.nn.relu(conv3+b3,name='relu3')#[?,7,7,128]

        with tf.name_scope('fc1') as scope:
            x_flat = tf.reshape(relu3,shape=[-1,7*7*128])#shape里最多有一个维度的值可以填写为-1，表示自动计算此维度
            w_fc1=tf.Variable(tf.truncated_normal(shape=[7*7*128,1024],stddev=0.05,mean=0),name='w_fc1')
            b_fc1 = tf.Variable(tf.zeros(1024),name='b_fc1')
            fc1 = tf.add(tf.matmul(x_flat,w_fc1),b_fc1)#[?,1024]
        with tf.name_scope('relu_fc1') as scope:
            relu_fc1 = tf.nn.relu(fc1,name='relu_fc1')
        with tf.name_scope('drop_1') as scope:
            drop_1 = tf.nn.dropout(relu_fc1,keep_prob=keep_prob,name='drop_1')
        with tf.name_scope('bn_fc1') as scope:
            bn_fc1 = tf.layers.batch_normalization(drop_1,name='bn_fc1')
            #y=γ(x-μ)/σ+β，其中x是输入，y是输出，μ是均值，σ是方差，γ和β是缩放、偏移系数
            #引入BN层的作用在于要最大限度的保证每次正向传播都输出在同一分布上，
            #这样反向计算时参照的数据样本分布就会和正向计算时的数据分布一样了。
            #保证了分布的统一，对权重的调整才会更有意义。

        with tf.name_scope('fc2') as scope:
            w_fc2 = tf.Variable(tf.truncated_normal(shape=[1024,512],stddev=0.05,mean=0),name='w_fc2')
            b_fc2 = tf.Variable(tf.zeros(512),name='b_fc2')
            fc2 = tf.add(tf.matmul(bn_fc1,w_fc2),b_fc2)
        with tf.name_scope('relu_fc2') as scope:
            relu_fc2 = tf.nn.relu(fc2,name='relu_fc2')
        with tf.name_scope('drop_2') as scope:
            drop_2 = tf.nn.dropout(relu_fc2,keep_prob=keep_prob,name='drop_2')
        with tf.name_scope('bn_fc2') as scope:
            bn_fc2 = tf.layers.batch_normalization(drop_2,name='bn_fc2')

        with tf.name_scope('fc3') as scope:
            w_fc3 = tf.Variable(tf.truncated_normal(shape=[512,2],stddev=0.05,mean=0),name='w_fc3')
            b_fc3 = tf.Variable(tf.zeros(2),name='b_fc3')
            fc3 = tf.add(tf.matmul(bn_fc2,w_fc3),b_fc3)
            tf.summary.histogram('fc3', fc3)
        return fc3#[?,2]

#训练参数
lr = 0.01#学习率
iterations = 101
batch_size = 64

#tf.variable_scope(): 可以让变量有相同的命名，允许创建新的variable并分享已创建的variable
#即使用 变量作用域(variable_scope) 来实现共享变量
with tf.variable_scope('input_x1') as scope:
    x1 = tf.placeholder(tf.float32, shape=[None, 784])#读入MNIST图像数组
    x_input_1 = tf.reshape(x1, [-1, 28, 28, 1])
with tf.variable_scope('input_x2') as scope:
    x2 = tf.placeholder(tf.float32, shape=[None, 784])
    x_input_2 = tf.reshape(x2, [-1, 28, 28, 1])
with tf.variable_scope('y') as scope:
    y = tf.placeholder(tf.float32, shape=[batch_size])

with tf.name_scope('keep_prob') as scope:
    keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope('siamese') as scope:
    out1 = siamese(x_input_1,keep_prob)#[batch_size,2]
    scope.reuse_variables()
    #tf.get_variable_scope().reuse_variables()是允许共享当前scope下的所有变量。reused_variables可以看同一个节点
    out2 = siamese(x_input_2,keep_prob)
    tf.summary.histogram('out1',out1)
    tf.summary.histogram('out2', out2)
with tf.variable_scope('metrics') as scope:
    loss = loss_with_step(out1, out2, y)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

loss_summary = tf.summary.scalar('loss',loss)#用tensorboard来显示标量信息
merged_summary = tf.summary.merge_all()#merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./SiameseNet_log',sess.graph) #tensorboard --logdir=SiameseNet_log
    sess.run(tf.global_variables_initializer())

    for itera in range(iterations):
        xs_1, ys_1 = mnist.train.next_batch(batch_size)#ys_1返回的是独热编码，需要进行转换，每次取batch_size个数据进行训练
        ys_1 = np.argmax(ys_1,axis=1)#取出对应轴元素最大值的下标，表示第几类
        xs_2, ys_2 = mnist.train.next_batch(batch_size)
        ys_2 = np.argmax(ys_2,axis=1)
        y_s = np.array(ys_1==ys_2,dtype=np.float32)#返回值为1. or 0.的数组（xs_1与xs_2是否是同类）
        _,train_loss,summ = sess.run([optimizer,loss,merged_summary],feed_dict={x1:xs_1,x2:xs_2,y:y_s,keep_prob:0.4})

        writer.add_summary(summ,itera)#tf.summary.FileWriter
        if itera % 10 == 0 :
            print('iter: {:5d} , train loss: {:5.4f}'.format(itera,train_loss))
    embed = sess.run(out1,feed_dict={x1:mnist.test.images,keep_prob:0.4})#return fc3=[batch_size,2]
    test_img = mnist.test.images.reshape([-1,28,28,1])
    writer.close()