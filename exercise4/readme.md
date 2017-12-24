
## 基本的CNN

我们这里的CNN用到了两个卷积层，一个隐藏层，一个输出层。

## 添加池化的CNN

实际上改动代码比较少，主要改动了CNN执行函数的前两层执行代码:

```
  def model(data):
    conv1 = tf.nn.relu(tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```

## 添加Dropout的CNN

这里我实际上做了三个工作：

* Dropout：在隐藏层的输出添加Dropout，实际上只有一行代码:

```
hidden = tf.nn.dropout(hidden, 0.5)  # Dropout 有一部分舍去 其他的加上
```

* 正则化:

```
 cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  loss = cross_entropy + 0.001 * (
          tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases) + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(
      layer4_biases))
```

* 学习率递减：

```
  learning_rate = tf.train.exponential_decay(1e-1, global_step, num_steps, 0.7, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
```

涉及到的exponential_decay这个函数，在[这里](http://zangbo.me/2017/07/01/TensorFlow_7/)有比较详细的解释

但是我感觉即使添加了这么多的优化，结果也没有好到哪去，这让我存在一些疑问

## 关键函数

### tf.nn.conv2d

tf.nn.conv2d是TensorFlow里面实现卷积的函数，参考文档对它的介绍并不是很详细，实际上这是搭建卷积神经网络比较核心的一个方法，非常重要

tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)

除去name参数用以指定该操作的name，与方法有关的一共五个参数：

第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一

第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维

第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4

第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）

第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true