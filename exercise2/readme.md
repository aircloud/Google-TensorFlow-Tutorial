
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
