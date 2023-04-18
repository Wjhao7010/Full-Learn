from functools import reduce

from webencodings import labels


class Perception(object):
    def __init__(self, input_num, activator):
        """
        这里函数的目的是初始化感知器，设置好输入参数的个数和激活函数
        激活函数的类型为double->double
        """
        self.activator = activator
        # 把权重wi向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项b初始化为0
        self.bias = 0.0

    def __str__(self):
        """
        打印出感知器所学习到的权重和偏置项
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        """
        输入向量，之后输出感知器的计算的结果
        """
        # 把input_vec[x1,x2,x3...xi]和weights[w1,w2,w3...wi]对应在一起
        # 变成[(x1*w1),(x2*w2),(x3*w3)...]
        # 对应好后使用map函数分别计算对应xi，wi的乘积
        # 最后用reduce求和
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda x, w: x * w,
                       zip(input_vec, self.weights)), 0.0) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        """
        输入对应的训练数据：一组向量、与每个向量对应的label；以及训练轮数和学习率
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        """
        通过一次迭代，把所有的训练数据都过一遍
        """
        # 把输入和输出打包在一起，成为样本的列表【（input_vecs, label),...]
        # 而每个循例那样本事(input_vec, label)
        samples = zip(input_vecs, labels)
        # 对每个样本都按照感知器的规则重新更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label ,rate):
        """
        按照感知器的规则更新权重
        """
        # 把input——vec[x1,x2,x3]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        self.weights = map(
            lambda x, w:w + rate * delta * x,
            zip(input_vec, self.weights)
        )
        # 更新bias
        self.bias += rate * delta

