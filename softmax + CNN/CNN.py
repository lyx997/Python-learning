# 导入mnist数据集
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# 导入tensorflow

import numpy as np
# 导入数据集   ## f.txt : 410x2400    f_label.txt : 410x10
training_data_value=np.loadtxt('f_value.txt') # [400x400]
training_data_label=np.loadtxt('f_test_label.txt') # [400x10]
test_value=np.loadtxt('f_test.txt')   # [400x400]
test_label=np.loadtxt('f_test_label.txt')  # [400x10]
new_value=np.loadtxt('fnew_.txt')   # [400x400]
new_label=np.loadtxt('f_new_label.txt')  # [400x10]
f1_value=np.loadtxt('f1.txt')   # [400x400]
f1_label=np.loadtxt('f1_label.txt')  # [400x10]
f2_value=np.loadtxt('f2.txt')   # [400x400]
f2_label=np.loadtxt('f2_label.txt')  # [400x10]
f3_value=np.loadtxt('f3.txt')   # [400x400]
f3_label=np.loadtxt('f3_label.txt')  # [400x10]
f4_value=np.loadtxt('f4.txt')   # [400x400]
f4_label=np.loadtxt('f4_label.txt')  # [400x10]
f5_value=np.loadtxt('f5.txt')   # [400x400]
f5_label=np.loadtxt('f5_label.txt')  # [400x10]

####################################模型构建#####################################
#构建输入x的占位符
x = tf.placeholder(tf.float32, [None, 2400])
#权重初始化：构建两个函数
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)  ### tf.truncated_normal() 有什么用？？
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#卷积和池化（vanilla版本）： 自己定义 边界、步长 ： 1步长（stride size），0边距（padding size）
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                        strides=[1, 1, 2, 1], padding='SAME')
with tf.device('/cpu:0'):
  ########## 第一层卷积： 卷积+ max pooling###################################################
  ###设置权重：卷积在每个5x5的patch中算出32个特征。   ？？？？如何得到32个
  W_conv1 = weight_variable([4, 4, 1, 8]) # 卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
  b_conv1 = bias_variable([8])
  ### 转化向量为图像，以实现卷积功能
  x_image = tf.reshape(x, [-1,6,400,1])  # 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数。 本图像为 6 x 400
  ###我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  ########## 第二层卷积： 为了构建一个更深的网络，我们会把几个类似的层堆叠起来。##############
  ###第二层中，每个5x5的patch会得到64个特征。   ？？？？如何得到64个
  W_conv2 = weight_variable([4, 4, 8, 16])  # 输入通道32，输出通道64
  b_conv2 = bias_variable([16])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  ########### 密集连接层 ######################################################################
  ###图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层
  W_fc1 = weight_variable([6 * 100 * 16, 1024])
  b_fc1 = bias_variable([1024])
  ### 将图片转化为一维向量
  h_pool2_flat = tf.reshape(h_pool2, [-1, 6*100*16])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  ############ dropout #########################################################################
  ### 为了减少过拟合，我们在输出层之前加入dropout。
  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  ############ 输出层 ##########################################################################
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  ##############################################################################################
  #################################### 训练与评估模型 #####################################
  y_ = tf.placeholder("float", [None,10])
  ### 用ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例
  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  ###############################################################################
  #初始化变量
  init = tf.global_variables_initializer()
  # 在一个Session里面启动我们的模型，并且初始化变量
  sess = tf.InteractiveSession()
  sess.run(init)
  # sess.run(tf.global_variables_initializer())
  ################################################################################
  for i in range(1000):
    training_data = np.hstack((training_data_value, training_data_label))
    new_data= np.hstack((new_value,new_label))
    training_data= np.vstack((training_data,new_data))
    np.random.shuffle(training_data)
    batch_xs_train, a1, a2 = np.array_split(training_data, (2400, 2401), axis=1)
    batch_ys_train = np.hstack((a1, a2))
    batch_xs_train = batch_xs_train[0:100, :]
    batch_ys_train = batch_ys_train[0:100, :]
    # batch = mnist.train.next_batch(50):
    train_accuracy = accuracy.eval(feed_dict={
          x:batch_xs_train, y_: batch_ys_train, keep_prob: 1.0})
    print("step %d, training accuracy train :%g"%(i, train_accuracy))
    print("test accuracy new :%g"%accuracy.eval(feed_dict={
      x: new_value , y_: new_label, keep_prob: 1.0}))
    print("test accuracy test :%g"%accuracy.eval(feed_dict={
      x: test_value , y_: test_label, keep_prob: 1.0}))
    print("test accuracy 按下/提起 :%g"%accuracy.eval(feed_dict={
      x: f1_value , y_: f1_label, keep_prob: 1.0}))
    print("test accuracy 五指收/张 :%g"%accuracy.eval(feed_dict={
      x: f2_value , y_: f2_label, keep_prob: 1.0}))
    print("test accuracy 向前/向后 :%g"%accuracy.eval(feed_dict={
      x: f3_value , y_: f3_label, keep_prob: 1.0}))
    print("test accuracy 向上/向下 :%g"%accuracy.eval(feed_dict={
      x: f4_value , y_: f4_label, keep_prob: 1.0}))
    print("test accuracy 向左/向右 :%g"%accuracy.eval(feed_dict={
      x: f5_value , y_: f5_label, keep_prob: 1.0}))
    train_step.run(feed_dict={x: batch_xs_train, y_: batch_ys_train, keep_prob: 0.5})


