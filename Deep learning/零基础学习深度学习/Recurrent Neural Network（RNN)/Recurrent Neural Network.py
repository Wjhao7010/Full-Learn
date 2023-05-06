import numpy as np

from CNN import ReluActivator, IdentityActivator, element_wise_op


class RecurrentLayer(object):
    def __init__(self, input_width, state_width,
                 activator, learning_rate):
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        self.times = 0  # 当前时刻初始化为t0
        self.state_list = []  # 保存各个时刻的state
        self.state_list.append(np.zeros((state_width, 1)))  # 初始化s0
        self.U = np.random.uniform(-1e-4, 1e-4,
                                   (state_width, input_width))  # 初始化U
        self.W = np.random.uniform(-1e-4, 1e-4,
                                   (state_width, state_width))  # 初始化W

    def forward(self, input_array):
        """
        根据式2进行前向计算
        """
        self.times += 1
        state = (np.dot(self.U, input_array) +
                 np.dot(self.W, self.state_list[-1]))
        # np.dot 向量点积和矩阵乘法
        element_wise_op(state, self.activator.forward)
        self.state_list.append(state)
        # 实现了对numpy数组进行按元素操作，并且把返回值写回数组中
        
