# train.py
import sys
import hugectr
from mpi4py import MPI

def train(json_config_file):
    solver_config = hugectr.solver_parser_helper(batchsize = 16384,
                                                 batchsize_eval = 16384,
                                                 vvgpu = [[0,1,2,3,4,5,6,7]],
                                                 repeat_dataset = True)
    sess = hugectr.Session(solver_config, json_config_file)
    sess.start_data_reading()
    for i in range(10000):
        sess.train()
        if (i % 100 == 0):
            loss = sess.get_current_loss()

if __name__ == "__main__":
    json_config_file = sys.argv[1]
    train(json_config_file)

#来自v3.0的readme
# 0x01 总体流程
# 1.1 概述
# HugeCTR 训练的过程可以看作是数据并行+模型并行。
#
# 数据并行是：每张 GPU卡可以同时读取不同的数据来做训练。
# 模型并行是：Sparse 参数可以被分布式存储到不同 GPU，不同 Node 之上，每个 GPU 分配部分 Sparse 参数。
# 训练流程如下：
#
# 首先构建三级流水线，初始化模型网络。初始化参数和优化器状态。
#
# Reader 会从数据集加载一个 batch 的数据，放入 Host 内存之中。
#
# 开始解析数据，得到 sparse 参数，dense 参数，label 等等。
#
# 嵌入层进行前向传播，即从参数服务器读取 embedding，进行处理。
#
# 对于网络层进行前向传播和后向传播，具体区分是多卡，单卡，多机，单机等。
#
# 嵌入层反向操作。
#
# 多卡之间交换 dense 参数的梯度。
#
# 嵌入层更新 sparse 参数。就是把反向计算得到的参数梯度推送到参数服务器，由参数服务器根据梯度更新参数。
#
# 1.2 如何调用
# 我们从一个例子中可以看到，总体逻辑和单机很像，就是解析配置，使用 session 来读取数据，训练等等，其中 vvgpu 是 device map。