# 本次是实现一个全连接神经网络
# 创建一个节点类，用来维护和记录自身信息（输出值a或者误差项等），以及这个节点相关上下游的连接
from functools import reduce
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Node(object):
    def __init__(self, layer_index, node_index):
        """
        构造节点对象
        layer_index：节点所属的层的编号
        node_index：节点编号
        """
        self.delta = None  # 后续添加
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []  # 下游
        self.upstream = []  # 上游
        self.output = 0
        self.data = 0

    def set_output(self, output):
        """
        set 设置节点的output值。如果节点是输入层节点就会使用到设置输出函数
        """
        self.output = output

    def append_downstream_connection(self, conn):
        """
        添加一个到下游节点的连接
        """
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        """
        添加一个到上游的连接
        """
        self.upstream.append(conn)

    def calc_output(self):
        """
        根据式1计算节点的输出  y=sigmoid(wT.x) 式1
        sigmoid(x) = 1/(1+e^-x)
        """
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        """
        节点属于隐藏层时，根据式4计算delta = ai(1-ai)求和 wki*delta ki
        """
        downstream_delta = reduce(  # 计算delta
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        """
        节点属输出层的话，根据式3计算delta = yi*（1-yi）*（ti-yi）
        """
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        """
        打印节点的信息
        """
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(object):
    # 创建一个ConstNode对象，为了实现一个输出恒为1的节点（计算偏置项wb时候需要）
    def __init__(self, layer_index, node_index):
        """
        创建节点对象
        layer_index:节点所属层的编号
        node_index:节点的编号
        """
        self.delta = None  # 后续添加
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        """
        添加一个到下游节点的连接
        """
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        """
        节点属于隐藏层的时，根据式4计算delta = ai(1-ai)求和 wki*delta ki
        """
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        """
        打印节点的信息
        """
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str

