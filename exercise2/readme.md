
## 任务

使用梯度下降和随机梯度下降训练一个全连接网络

本次练习包含两个部分，采用梯度下降和随机梯度下降来训练一个全链接网络。其中`1-GradientDescent.py`是采用梯度下降，而`2-GradientStochasticDescent.py`采用的是随机梯度下降。

随机梯度下降每次只是选用一部分训练集，因此可以提高速度。

## 思考

在我看来，要初步了解神经网络，我们首先要知道这个训练过程训练的是什么，我认为，实际上训练的是参数，这个参数用于得出结果。

比如这里的28*28图像数据集的十分类问题，我们训练一个(784,10)维度的矩阵，然后还有一个(1,10)的biases(可以理解为y=ax+b中的b，只不过这里的a、x、b都是矩阵)


## 关键函数

### softmax_cross_entropy_with_logits

这个函数看名字都知道，是将稀疏表示的label与输出层计算出来结果做对比，函数的形式和参数如下：

nn.sparse_softmax_cross_entropy_with_logits(logits,label,name=None)


第一个坑:logits表示从最后一个隐藏层线性变换输出的结果！假设类别数目为10，那么对于每个样本这个logits应该是个10维的向量，且没有经过归一化，所有这个向量的元素和不为1。然后这个函数会先将logits进行softmax归一化，然后与label表示的onehot向量比较，计算交叉熵。
也就是说，这个函数执行了三步（这里意思一下）：

sm=nn.softmax(logits)
onehot=tf.sparse_to_dense(label，…)
nn.sparse_cross_entropy(sm,onehot)

第二个坑:输入的label是稀疏表示的，就是是一个[0，10）的一个整数，这我们都知道。但是这个数必须是一维的！就是说，每个样本的期望类别只有一个，属于A类就不能属于其他类了。
