import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np
import tensorflow as tf
training_fft =np.loadtxt('value.txt')
training_label=np.loadtxt('label.txt')



# state = tf.Variable(0, name="counter")
# input1 = tf.placeholder(tf.float32,[None, 5000])
# e1=input1
# input2 = tf.constant(2.0)
# output = tf.mul(input1,input2)
# one = tf.constant(1)
# new_value = tf.add(state, one
# update = tf.assign(state, new_value)
arr1 = np.arange(9).reshape(3,3)
arr2 = np.arange(15).reshape(3,5)
arr=np.hstack((arr1,arr2))
arr_,arr_1,arr_2=np.array_split(arr,(3,5),axis=1)
arr_3=np.hstack((arr_1,arr_2))
training_data = np.hstack((training_fft, training_label))
np.random.shuffle(training_data)
training_fft1, a1, a2 = np.array_split(training_data, (400, 401), axis=1)
training_label1 = np.hstack((a1, a2))
b = tf.argmax(training_label1[1:3], 1)
init = tf.global_variables_initializer()

with tf.Session() as sess:
 sess.run(init)

 # print(sess.run(state))# 打印初值
 # for _ in range(3):
 #   sess.run(update)
 #   print(state.eval()) # 记住代码要对齐
 # for i in range(2):
 #  print(i,training_data[14,74])
 #  print(i,label[2])  # 取ceshi数组的第3行，即 ceshi[2]
 #  print(i,label[0,0:10])
 print(training_label1[1:3])
 print(sess.run(b))
 # print(training_label[1])
 # for i in range(3):
 #  print(training_label1[i])
 #  print(training_data.shape)
 #  print(training_fft1.shape)
 #  print(training_label1.shape)

## tf.argmax() 例子
# a=tf.get_variable(name='a',
#                   shape=[3,4],
#                   dtype=tf.float32,
#                   initializer=tf.random_uniform_initializer(minval=-1,maxval=1))
# b=tf.argmax(input=a,dimension=0)
# c=tf.argmax(input=a,dimension=1)
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# print(sess.run(a))
# print(sess.run(b))
# print(sess.run(c))