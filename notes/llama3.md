# llama 3.1

- dense transformer 405B（有一种说法是 dense 个人电脑的显存才放得下，Mixture of Expert就放不下了）70B 其实效果和 405B 比较接近了
- 128k token context window
- 15 T multilingual tokens，我们可以用 common crawl 来做抓取
- Post Training 用的比较传统的 supervised finetuning (SFT), rejection sampling (RS), direct preference optimization(DPO)，没有用 RLHF
- data cleaning: removing domain with Personally Identifiable Information, removing domain with adult content
- customize HTML parser, 尤其是 math（使用的公式图片的alt，假设alt就是公式内容） & codes，去除所有的markdown markers
- de-duplication: url, document (MinHash), line-level（使用了30M文本中超过6次的内容就会被删除，这两个数字应该是权衡成本的超参数，主要是去除掉一些 boilerplate，会有很大的提升）
- data mix：使用 LLM 来给数据分成几类，然后不同的数据的重要性是不一样的（比如说娱乐的信息对于模型学习就没什么帮助）llama的比例大概是通用知识50%，数学推理25%，17%代码，8%多语言token
- annealing data/ fine tuning：考前突击一下，把学习率稍微打开一点然后再慢慢调低

# Shingling + MinHash + Locality Sensitive Hashing

- 如何压缩特征：shingling，把文本按照固定长度断开
- 如何比较海量数据的相似度：minhash，用0/1矩阵表达一个元素是否拥有某一个feature（类似于one-hot encoding，区别在于现在可以有多个1，feature之间不是相斥的）
  - 然后我们randomize所有的feature的顺序，计算每一个元素第一个非0的feature是序号几
  - 我们可以用这个来压缩上面那个0/1矩阵，变成每一个元素在不同的permutation中最小的1序号
  - 有一个非常重要的定理：Jaccard Similarity(s1, s2) = p(m(s1), m(s2))，也符合直觉，因为最小序号也需要是1，所以分母是交集，而相等的时候就是并集。我们的permutation越多，得到的结果就越近似
- Locality Sensitive Hashing 是类似于 bloom filter，只有碰撞了的时候才去计算这两个文档的minhash相似性

# Heuristic filtering

- n-gram coverage ratio：移除同一行里反复出现的内容，一般是 logging 或者 error message
- dirty word/ curse
- token distribution Kullback-Leibler divergence：如果一个 document 的内容和其他的 document 非常不一样，那么这可能是一个 outlier，会影响学习的效果
- 可以用LLM来给 document 打分和打tag，可以丢弃质量很差的内容，打了tag 的内容也可以之后用于特殊的训练，用LLM是很贵的做法（主要是机器，前面主要是人力很贵）

# 模型

- Group Query Attention GQA，主要是减少 decoding 的内存需要（主要是可以共用KVCache，多头共用的程度越高，那么效率越高，但是效果越差，所以一般两两合并，折中方案）
- Mask Attention，同一个sequence里面不同的document之间不算score，只算document内部的。如果sequence很长的时候就很有用
- Token vocabulary变成128K，每一个token所代表的char也增加了（所以同一段话现在需要更少的token来表示了）
- scaling law 一般衡量的是预测下一个词的 loss，这也是模型训练和验证过程中用到的，但是我们实际上需要衡量的是整体的任务完成情况
  - llama 构建了下游任务 down-stream tasks 的 negative log-likelihood 和 training FLOPs 之间的关系（这个东西改模型架构就要重新算）
  - 找到下游任务 down-stream tasks 的 negative log-likelihood 和 task accuracy 之间的关系（这个和模型架构无关，可以复用之前的 llama 2 模型的数据）
  - 有了这些数据之后，我们就可以预测当前算力和一定的时间内的FLOPs所对应的合适的training token的数量（不能比这个模型小太多）

# Infra

- 16000 H100 不稳定
- 240 PB SSD，训练一次用1PB很正常，文本数据1PB，多模态100PB也很正常，每一个step都会存到这个分布式的系统
- 大模型用的 RDMA over Converged Ethernet，小模型实验的时候用的是Nvidia Infiniband（不成熟）
- 一个rack只有两个机器16个GPU
- 192个rack连到同一个cluster switch，switch之上还有switch
- load balancer，congestion control 避免 buffer 溢出（deep buffer）
- parallel：网络每一层，参数，数据，sequence kvcache都可以切分：all-gather
- 数字稳定性：计算gradient的时候换回FP32而不是16

# training recipe

- pretraining
  - 8000 步线性 从 0 warm up 到 peak learning rate $8 \times 10^{-5}$，然后用 cos decay 一百二十万步到 $8 \times 10^{-7}$
  - 因为大模型训练过程中不稳定（有spike），所以采用了一开始 batch size 4M，sequence length 4096，然后变成 8M 和 8192。最后变成 16M
  - upsample 非英语数据，数学，在最后阶段学习最近的网页数据，downsample 低质量内容
  - long context pretraining：如何从 8k 上下文到 128k（直接部署的话 8k 乘4还可以，乘16就不太行了）
    - 但是长序列很难训练（拟合和分布式，这就是为什么infra需要有context维度上的分布式）
  - annealing：评测数据集最后练一下（高质量数据）线性降为0