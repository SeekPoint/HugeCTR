import hugectr
from mpi4py import MPI
solver = hugectr.CreateSolver(max_eval_batches = 300,
                              batchsize_eval = 16384,
                              batchsize = 16384,
                              lr = 0.001,
                              vvgpu = [[0]],
                              repeat_dataset = True)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                  source = ["./criteo_data/file_list.txt"],
                                  eval_source = "./criteo_data/file_list_test.txt",
                                  check_type = hugectr.Check_t.Sum)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                    update_type = hugectr.Update_t.Global,
                                    beta1 = 0.9,
                                    beta2 = 0.999,
                                    epsilon = 0.0000001)
# 之前提到了Parser是解析配置文件，HugeCTR 也支持代码设置，
# 比如下面就设定了两个DataReaderSparseParam，
# 也有对应的DistributedSlotSparseEmbeddingHash。

# 0x05 初始化
# 5.1 配置
# 前面提到了可以使用代码完成网络配置，我们从下面可以看到，DeepFM 一共有三个embedding层，
# 分别对应 wide_data 的 sparse 参数映射到dense vector，deep_data 的 sparse 参数映射到dense vector，。
'''
4.3.4 嵌入表大小
我们已经知道可以通过哈希表来进行缩减嵌入表大小，现在又知道其实还可以通过combine来继续化简，所以在已经有了哈希表基础之上，我们需要先问几个问题。
        目前 hash_table_value 究竟有多大？就是权重矩阵（稠密矩阵）究竟多大？
        embedding_feature （嵌入层前向传播的输出）究竟有多大？就是输出的规约之后的矩阵应该有多大？
        embedding_feature 的每一个元素是怎么计算出来的？
        实际矩阵有多大？
我们解答一下。
        第一个问题hash_table_value 究竟有多大？
前文之中有分析 hash_table_value 大小是：max_vocabulary_size_per_gpu_ = embedding_data_.embedding_params_.max_vocabulary_size_per_gpu;
实际上，大致可以认为，hash_table_value 的大小是：(value number in CSR) * (embedding_vec_size) 。
hash_table_value 的数值是随机初始化的。每一个原始的 CSR user ID 对应了其中的 embedding_vec_size 个元素。
hash_value_index 和 row_offset 凑在一起，
就可以找到每一个原始的 CSR user ID 对应了其中的 embedding_vec_size 个元素。
        第二个问题：embedding_feature 究竟有多大？就是逻辑上的稠密矩阵究竟有多大？从代码可以看到，
            embedding_feature[feature_row_index * embedding_vec_size + tid] =
            TypeConvertFunc<TypeEmbeddingComp, float>::convert(sum);
可见，embedding_feature 的大小是：(row number in CSR) * (embedding_vec_size) 。因此，对于 embedding_feature_tensors_，我们抽象一下，输入假设是4行 CSR格式，则输出就是4行稠密向量格式。
        第三个问题：embedding_feature 的每一个元素是怎么计算出来的？
是遍历slot和element，进行计算。
            sum += (value_index != std::numeric_limits<size_t>::max())
            ? hash_table_value[value_index * embedding_vec_size + tid]
            : 0.0f;
        第四个问题：实际embedding矩阵，或者说工程上的稠密矩阵有多大？
其实就是 slot_num * embedding_vec_size。row number 其实就是 slot_num。从下面输出可以看到。
以 deep_data 为例，其slot num 是26，embedding_vec_size = 16，最后输出的一条样本大小是 [26 x 16]。

...
输出：

"------------------------------------------------------------------------------------------------------------------\n",
"Layer Type                              Input Name                    Output Name                   Output Shape \n",
"------------------------------------------------------------------------------------------------------------------\n",
"DistributedSlotSparseEmbeddingHash      wide_data                     sparse_embedding2             (None, 1, 1)  \n",
"DistributedSlotSparseEmbeddingHash      deep_data                     sparse_embedding1             (None, 26, 16)"\
'''
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        [hugectr.DataReaderSparseParam("wide_data", 30, True, 1),
                        hugectr.DataReaderSparseParam("deep_data", 2, False, 26)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 23,
                            embedding_vec_size = 1,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding2",
                            bottom_name = "wide_data",
                            optimizer = optimizer))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 358,
                            embedding_vec_size = 16,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "deep_data",
                            optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding1"],
                            top_names = ["reshape1"],
                            leading_dim=416))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding2"],
                            top_names = ["reshape2"],
                            leading_dim=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                            bottom_names = ["reshape1", "dense"],
                            top_names = ["concat1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["concat1"],
                            top_names = ["fc1"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc1"],
                            top_names = ["relu1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu1"],
                            top_names = ["dropout1"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout1"],
                            top_names = ["fc2"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc2"],
                            top_names = ["relu2"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu2"],
                            top_names = ["dropout2"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout2"],
                            top_names = ["fc3"],
                            num_output=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Add,
                            bottom_names = ["fc3", "reshape2"],
                            top_names = ["add1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["add1", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()
model.fit(max_iter = 2300, display = 200, eval_interval = 1000, snapshot = 1000000, snapshot_prefix = "wdl")
