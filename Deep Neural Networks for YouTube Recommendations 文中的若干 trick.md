# Deep Neural Networks for YouTube Recommendations 文中的若干 trick

> 《Deep Neural Networks for YouTube Recommendations》 是 Youtube 团队 2016 年发表在 RecSys 会议的一篇论文，主要讲了在实际业务中对 DNN 的应用。但除了 DNN 的部分之外，这篇文章也充满了许多值得在实践中借鉴参考的经验。

## Matching 阶段

### 问题建模

Matching 阶段建模为超大规模多分类问题，模型为一个 softmax多分类器。

模型训练阶段输出层为 softmax，serving 阶段直接拿 user vector 查询 item，考虑到性能，nearest neighbor 使用的是局部敏感哈希

user vector 为 softmax 层的输入，item matrix 为 softmax 层的权重。

### 重要特征

1. **抛弃 session 时序信息** ：session 类的特征（如历史搜索 query 分词后的 token，历史播放 item 列表）都是单独 embedding 后再进行加权平均后得到一个 dense 的 vector，**不是很明白这样做的原因** 
2. **Example Age 特征** ，考虑 item 时效性
3. 抛弃 session 类特征中的穿越情况，使用 history predict next；CF 其实有穿越问题

### 样本工程

1. **为每个用户生成固定数量训练样本**：为每个用户固定样本数量上限，平等的对待每个用户，避免 loss 是由被少数 active 用户贡献，能明显提升线上效果
2. 使用跨域的日志数据作为样本源，为推荐场景提供 explore

### 训练过程

对优化目标采用 negative sampling 方法，每个训练样本的训练只会更新一小部分的模型权重，从而降低计算负担



## Ranking 阶段

### 问题建模

**对用户浏览时长建模可以解决一部分 clickbait 的问题**，训练时使用 weighted linear regression 作为输出层，其中正样本对 loss 的贡献为用户浏览时长，负样本对 loss 的贡献为 1；serving 时使用 $e^x$

### 特征工程

1. Ranking 阶段需要可以很好的合并多个召回源数据的能力，把召回阶段的信息传递到 Ranking 阶段同样能很好的提升效果，比如召回源和召回时的打分。
2. 不同维度下的相同 item ID 的 embedding vector 是共享的，可以加快训练速度，这些 itemID -> emb vector 向量的映射关系是提前训练好存储起来的，在实际模型训练前就已经可以获取每个 item 的 emb vector 了
3. 连续类特征归一化方法：$\overline{x}=\int_{-\infty}^{x}dx$
4. 加入连续类特征的 $x^2$ 项和 $\sqrt{x}$ 项，来引入非线性化