from functools import reduce


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
            reduce(lambda a, b:a+b,
                map(lambda x, w:x*w,
                    zip(input_vec, self.weights)))) + self.