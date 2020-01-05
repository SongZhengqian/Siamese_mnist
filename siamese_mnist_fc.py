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
def siamese(inputs):#inputs=[?,28,28,1]
    with tf.name_scope('fc1') as scope:
        x_flat = tf.reshape(inputs,shape=[-1,28*28])#shape里最多有一个维度的值可以填写为-1，表示自动计算此维度
        w_fc1=tf.Variable(tf.truncated_normal(shape=[28*28,100],stddev=0.05,mean=0),name='w_fc1')
        b_fc1 = tf.Variable(tf.zeros(100),name='b_fc1')
        tf.summary.histogram('w_fc1', w_fc1)
        tf.summary.histogram('b_fc1', b_fc1)
        fc1 = tf.add(tf.matmul(x_flat,w_fc1),b_fc1)#[?,100]
    with tf.name_scope('fc2') as scope:
        w_fc2 = tf.Variable(tf.truncated_normal(shape=[100,10],stddev=0.05,mean=0),name='w_fc2')
        b_fc2 = tf.Variable(tf.zeros(10),name='b_fc2')
        tf.summary.histogram('w_fc2', w_fc2)
        tf.summary.histogram('b_fc2', b_fc2)
        fc2 = tf.add(tf.matmul(fc1,w_fc2),b_fc2)
    with tf.name_scope('fc3') as scope:
        w_fc3 = tf.Variable(tf.truncated_normal(shape=[10,2],stddev=0.05,mean=0),name='w_fc3')
        b_fc3 = tf.Variable(tf.zeros(2),name='b_fc3')
        tf.summary.histogram('w_fc3', w_fc3)
        tf.summary.histogram('b_fc3', b_fc3)
        fc3 = tf.add(tf.matmul(fc2,w_fc3),b_fc3)
    return fc3#[?,2]

#训练参数
lr = 0.01#学习率
iterations = 500
batch_size = 64

#tf.variable_scope(): 可以让变量有相同的命名，允许创建新的variable并分享已创建的variable
#即使用变量作用域(variable_scope) 来实现共享变量
with tf.variable_scope('input_x1') as scope:
    x1 = tf.placeholder(tf.float32, shape=[None, 784])#读入MNIST图像数组
    x_input_1 = tf.reshape(x1, [-1, 28, 28, 1])
with tf.variable_scope('input_x2') as scope:
    x2 = tf.placeholder(tf.float32, shape=[None, 784])
    x_input_2 = tf.reshape(x2, [-1, 28, 28, 1])
with tf.variable_scope('y') as scope:
    y = tf.placeholder(tf.float32, shape=[batch_size])

with tf.variable_scope('siamese') as scope:
    out1 = siamese(x_input_1)#[batch_size,2]
    scope.reuse_variables()
    #tf.get_variable_scope().reuse_variables()是允许共享当前scope下的所有变量。reused_variables可以看同一个节点
    out2 = siamese(x_input_2)
    tf.summary.histogram('out1',out1)
    tf.summary.histogram('out2', out2)
with tf.variable_scope('metrics') as scope:
    loss = loss_with_step(out1, out2, y)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

loss_summary = tf.summary.scalar('loss',loss)#用tensorboard来显示标量信息
merged_summary = tf.summary.merge_all()#merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示

saver = tf.train.Saver()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./SiameseNet_log',sess.graph) #tensorboard --logdir=SiameseNet_log
    sess.run(tf.global_variables_initializer())
    save_path = saver.save(sess,'C:\\Users\\Y\\Desktop\\model\\siamese_net.ckpt')
    print("Save to path:",save_path)
    for itera in range(iterations):
        xs_1, ys_1 = mnist.train.next_batch(batch_size)#ys_1返回的是独热编码，需要进行转换，每次取batch_size个数据进行训练
        ys_1 = np.argmax(ys_1,axis=1)#取出对应轴元素最大值的下标，表示第几类
        xs_2, ys_2 = mnist.train.next_batch(batch_size)
        ys_2 = np.argmax(ys_2,axis=1)
        y_s = np.array(ys_1==ys_2,dtype=np.float32)#返回值为1. or 0.的数组（xs_1与xs_2是否是同类）
        _,train_loss,summ = sess.run([optimizer,loss,merged_summary],feed_dict={x1:xs_1,x2:xs_2,y:y_s})

        writer.add_summary(summ,itera)#tf.summary.FileWriter
        if itera % 100 == 0 :
            print('iter: {:5d} , train loss: {:5.4f}'.format(itera,train_loss))
    embed = sess.run(out1,feed_dict={x1:mnist.test.images})#return fc3=[batch_size,2]
    test_img = mnist.test.images.reshape([-1,28,28,1])
    writer.close()

