from perceptron import Perceptron
# 监督学习类型  运用员工已知工龄和工资来训练线性单元
# 继承了Perceptron
# 定义激活函数f
f = lambda x: x


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, f)


def get_training_dataset():
    """
    创建5个人的收入数据
    """
    # 构建训练数据
    # 输入向量列表，每项是工作年限
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，月薪要与输入一一对应
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    # 使用数据训练线性单元
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = LinearUnit(1)
    # 训练并且迭代10轮, 学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    # 返回训练好的线性单元
    return lu


if __name__ == '__main__':
    """
    训练线性单元
    """
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print(linear_unit)
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
