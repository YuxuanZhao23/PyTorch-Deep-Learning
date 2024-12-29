# llama 3.1

- dense transformer 405B（有一种说法是 dense 个人电脑的显存才放得下，Mixture of Expert就放不下了）70B 其实效果和 405B 比较接近了
- 128k token context window
- 15 T multilingual tokens，我们可以用 common crawl 来做抓取
- Post Training 用的比较传统的 supervised finetuning (SFT), rejection sampling (RS), direct preference optimization(DPO)，没有用 RLHF
- data cleaning: removing domain with Personally Identifiable Information, removing domain with adult content
- customize HTML parser, 尤其是 math（使用的公式图片的alt，假设alt就是公式内容） & codes，去除所有的markdown markers
- de-duplication: url, document (MinHash), line-level

# Shingling + MinHash + Locality Sensitive Hashing

- 如何压缩特征：shingling，把文本按照固定长度断开
- 如何比较海量数据的相似度：minhash，用0/1矩阵表达一个元素是否拥有某一个feature（类似于one-hot encoding，区别在于现在可以有多个1，feature之间不是相斥的）
  - 然后我们randomize所有的feature的顺序，计算每一个元素第一个非0的feature是序号几
  - 我们可以用这个来压缩上面那个0/1矩阵，变成每一个元素在不同的permutation中最小的1序号
  - 有一个非常重要的定理：Jaccard Similarity(s1, s2) = p(m(s1), m(s2))，也符合直觉，因为最小序号也需要是1，所以分母是交集，而相等的时候就是并集。我们的permutation越多，得到的结果就越近似
- Locality Sensitive Hashing 是类似于 bloom filter，只有碰撞了的时候才去计算这两个文档的minhash相似性