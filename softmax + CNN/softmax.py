## 使用tensorflow搭建softmax分类器

########################################模型构建####################################
# 导入tensorflow
import tensorflow as tf
import numpy as np
# 导入数据集
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

# x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。 （这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder(tf.float32, [None, 2400])
#  一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。
# 对于各种机器学习应用，一般都会有模型参数，可以用Variable表示。
W = tf.Variable(tf.zeros([2400,10]))  # 代表权重Wi ；
b = tf.Variable(tf.zeros([10]))      # 代表偏置量Bi  因为W和B的值需要学习来确定，所以初始值设为0
# 实现模型，y为由Xs计算得到的
y = tf.nn.softmax(tf.matmul(x,W) + b)
# 为了计算交叉熵，用一个新的占位符placeholder表示用于输入正确值（作为一个输入接口），y_是实际的分布（我们输入的one-hot vector)，即标签ys
y_ = tf.placeholder("float", [None, 10])
# 计算交叉熵，我们的损失函数是目标类别和预测类别之间的交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y+1e-10))
# 最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
#####################################变量初始化###########################
#初始化变量
init = tf.global_variables_initializer()
# 在一个Session里面启动我们的模型，并且初始化变量
sess = tf.InteractiveSession()
sess.run(init)
###############################评估模型（测试集）###################################
# tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测
# 是否真实标签匹配(索引位置一样表示匹配)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
#例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#计算所学习到的模型在测试数据集上面的正确率
##############################训练模型##############################################
# 开始训练模型，这里我们让模型循环训练1000次，[60000，784]
#该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据样本，然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
#使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练。
for i in range(1000):
  # batch_xs, batch_ys = mnist.train.next_batch(100)  # DataSet.next_batch()是用于获取以batch_size为大小的一个元组，其中包含了一组图片和标签
  # batch_xs = training_data_value[i*10:i*10+10]
  # batch_ys = training_data_label[i*10:i*10+10]
  training_data = np.hstack((training_data_value, training_data_label))
  new_data= np.hstack((new_value,new_label))
  training_data= np.vstack((training_data,new_data))
  np.random.shuffle(training_data)
  batch_xs_train, a1, a2 = np.array_split(training_data, (2400, 2401), axis=1)
  batch_ys_train = np.hstack((a1, a2))
  batch_xs_train=batch_xs_train[0:50,:]
  batch_ys_train=batch_ys_train[0:50,:]
  sess.run(train_step, feed_dict={x: batch_xs_train, y_: batch_ys_train})  # feed的作用：给x，y_赋值；
  # if i%50==0:
  print('step',i,'accurcy test:', sess.run(accuracy, feed_dict={x: test_value, y_: test_label}))  # 把测试集所有数据输入
  print('step',i,'accurcy new:', sess.run(accuracy, feed_dict={x: new_value, y_: new_label}))  # 把测试集所有数据输入
  print('step', i, 'accurcy f1:', sess.run(accuracy, feed_dict={x: f1_value, y_: f1_label}))  # 把测试集所有数据输入
  print('step', i, 'accurcy f2:', sess.run(accuracy, feed_dict={x: f2_value, y_: f2_label}))  # 把测试集所有数据输入
  print('step', i, 'accurcy f3:', sess.run(accuracy, feed_dict={x: f3_value, y_: f3_label}))  # 把测试集所有数据输入
  print('step', i, 'accurcy f4:', sess.run(accuracy, feed_dict={x: f4_value, y_: f4_label}))  # 把测试集所有数据输入
  print('step', i, 'accurcy f5:', sess.run(accuracy, feed_dict={x: f5_value, y_: f5_label}))  # 把测试集所有数据输入
# for i in range(1000):
#     training_data = np.hstack((new_value, new_label))
#     np.random.shuffle(training_data)
#     batch_xs_train, a1, a2 = np.array_split(training_data, (2400, 2401), axis=1)
#     batch_ys_train = np.hstack((a1, a2))
#     batch_xs_train = batch_xs_train[0:50, :]
#     batch_ys_train = batch_ys_train[0:50, :]
#     # batch = mnist.train.next_batch(50)
#     if i % 100 == 0:
#       train_accuracy = accuracy.eval(feed_dict={
#         x: batch_xs_train, y_: batch_ys_train})
#       print("step %d, training accuracy %g" % (i, train_accuracy))
#     train_step.run(feed_dict={x: batch_xs_train, y_: batch_ys_train})
#
# print("test accuracy %g" % accuracy.eval(feed_dict={
#     x: new_value, y_: new_label}))
# print("test accuracy %g" % accuracy.eval(feed_dict={
#     x: test_value, y_: test_label}))
# np.savetxt("b_new.txt", sess.run(b));
# np.savetxt("w_new.txt", sess.run(W));

   # if i%40==39:
  # print('step',i, 'accurcy', sess.run(accuracy, feed_dict={x: batch_xs_train, y_: batch_ys_train}))  # 查看每次训练时准确率
  # print('step', i, 'y', sess.run(y, feed_dict={x: batch_xs, y_: batch_ys}))  # 把测试集所有数据输入
  # batch_xs = training_data_value[0:10]
  # batch_ys = training_data_label[0:10]
  # print('step',i, 'accuracy', sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))  # 把测试集所有数据输入

#####################################进行评估###################################
# for j in range(40):
#  # test_data_value=training_data_value[i*10:i*10+10]
#  # test_data_label=training_data_label[i*10:i*10+10]
#  test_data = np.hstack((training_data_value, training_data_label))
#  np.random.shuffle(test_data)
#  batch_xs_test, a1, a2 = np.array_split(test_data, (400, 401), axis=1)
#  batch_ys_test = np.hstack((a1, a2))
#  batch_xs_test = batch_xs_test[0:40]
#  batch_ys_test = batch_ys_test[0:40]
#  print('step',j,'accurcy',sess.run(accuracy,feed_dict={x: batch_xs_test, y_: batch_ys_test})) # 把测试集所有数据输入
 # print('step', j, 'y', sess.run(y[1], feed_dict={x: batch_xs_test, y_: batch_ys_test}))  # 把测试集所有数据输入


   # a=sess.run(y, feed_dict={x: test_value, y_: test_label})
   # print("y",a )