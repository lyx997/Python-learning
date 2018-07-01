import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
########################################模型构建####################################
# 导入tensorflow
import tensorflow as tf
test_images=mnist.test.images;
test_labels=mnist.test.labels;
# x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。 （这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder(tf.float32, [None, 784])
#  一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。
# 对于各种机器学习应用，一般都会有模型参数，可以用Variable表示。
W = tf.Variable(tf.zeros([784,10]))  # 代表权重Wi ； zero应该为创建全为 0 的矩阵； W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。
b = tf.Variable(tf.zeros([10]))      # 代表偏置量Bi  因为W和B的值需要学习来确定，所以初始值设为0
# 实现模型，y为由Xs计算得到的
y = tf.nn.softmax(tf.matmul(x,W) + b)
# 为了计算交叉熵，用一个新的占位符placeholder表示用于输入正确值（作为一个输入接口），y_是实际的分布（我们输入的one-hot vector)，即标签ys
y_ = tf.placeholder("float", [None,10])
# 计算交叉熵，我们的损失函数是目标类别和预测类别之间的交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


#####################################至此，完成模型的设置###########################
#初始化变量
init = tf.global_variables_initializer()
# 在一个Session里面启动我们的模型，并且初始化变量
sess = tf.Session()
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
for i in range(5000): # range(x), x是训练的次数
  batch_xs, batch_ys = mnist.train.next_batch(100)  # DataSet.next_batch()是用于获取以batch_size为大小的一个元组，其中包含了一组图片和标签
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # feed的作用：给x，y_赋值；
  if i%50==0:
    print('step', i, 'accurcy', sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))  # 把测试集所有数据输入

print(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels})) # 把测试集所有数据输入
