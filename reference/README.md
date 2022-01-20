# Contrastive Learning

+ **Momentum Contrast for Unsupervised Visual Representation Learning**
> 从字典查询的角度来理解对比学习，将查询样本和键值样本分别通过各自的编码器生成查询和键值，同一样本通过加入噪声或图片裁剪后构成相似样本对，不同样本之间构成不相似样本对，在匹配查询和键值时需尽可能使得相似样本对的损失小，而不相似样本对的损失大，为了使得在每次匹配的过程中增大字典键值数，通过构成键值序列，使其不受限于mini-batch的大小，每个batch将样本加入键值序列的同时弹出序列中最早加入的样本。

+ **Dimensionality Reduction by Learning an Invariant Mapping**
> 基于降维后相似样本靠近，不相似样本远离的思想，构造了对比损失函数，当降维后不相似样本之间距离较远时则不产生损失，通过构造每个样本的相似样本集和不相似样本集即可对模型进行训练。

+ **A Simple Framework for Contrastive Learning of Visual Representations**
> 每个样本通过两种不同的数据增强方式得到一对相似样本，这对相似样本经过一次映射生成特征表示空间，再经过一次映射生成对比损失计算空间，最终最大化对比损失计算空间内相似样本的余弦夹角，特征表示空间内的特征用于下游任务。文中通过大量实验证明了通过对比损失计算空间的生成，而非直接在特征表示空间内计算对比损失能够提升模型性能，同时更强的数据增强方式以及批次大小和轮次数的增加也能够提升模型性能。

+ **Representation Learning with Contrastive Predictive Coding**
> 利用数据序列之间的预测关系实现对比学习，并非直接对未来时刻的原始特征进行预测，而是最大化和未来时刻编码特征的互信息，较适用于具有时序性的数据，如语音信号。

+ **Supervised Contrastive Learning**
> 由于自监督对比学习采用其他样本作为不相似样本，其中可能会包含同类别的相似样本，造成错误的特征表示学习，因此提出有监督的对比学习，在已知标签的情况下将同一类别中的样本作为相似样本，不同类别中的样本作为不相似样本，使得在学习到的特征表示空间中类内紧缩类间分离，能够同时包含多个相似样本进行训练，本质上可以看做是一种对交叉熵损失函数的改进。

+ **Deep Clustering for Unsupervised Learning of Visual Features**
> 采用聚类方法对特征表示进行无监督学习，首先利用聚类算法给出特征表示的伪标签，基于伪标签利用分类器对网络进行反向传播训练，最终将特征表示用于下游任务。

+ **Semi-Supervised Learning with Ladder Networks**
> 一种半监督学习的框架，主要思想是将自编码器产生的无监督损失和隐含特征用于有监督的损失一起用于训练，不同于一般的抗噪自编码器，模型在每一层都加入了噪声以提升鲁棒性，并且在重构时使用了跳跃连接，避免由编码造成的信息损失。

+ **MixMatch: A Holistic Approach to Semi-Supervised Learning**
> 一种集合了现有常用半监督思想的半监督学习方法，包括了数据增强、熵最小化（通过标签的锋利化实现）、一致性假设（同一样本的不同增强方式后应保留相同的标签）、普通正则化项、MixUp（样本和标签采用同样的线性组合方式生成新样本）。

+ **A Survey on Semi-, Self- and Unsupervised Learning for Image Classification**
> 半监督、自监督和无监督学习综述，介绍了常用半监督、自监督和无监督学习的共通思想，以及25种具体算法，作者认为无监督学习未来可能会逐渐被弃用。

+ **EnAET: Self-Trained Ensemble AutoEncoding Transformations for Semi-Supervised Learning**
> 将一种自监督学习方法AET（Auto-Encoding Transformations，预测图像的变换参数）引入半监督框架，模型Loss由半监督学习Loss（此处采用MixMatch）、原始图像和变换后图像预测结果之间的KL散度（约束预测一致性）和AET的Loss（通过集成的方式实现空间和非空间的多种变换参数的预测）共同构成。

+ **Interpolation Consistency Training for Semi-Supervised Learning**
> 基于插值一致性实现半监督学习，有标签样本采用常规的交叉熵产生Loss，无标签样本采用两个样本的线性组合作为输入，伪标签的线性组合作为期望输出，利用二范数产生无标签Loss。

+ **Averaging Weights Leads to Wider Optima and Better Generalization**
> 通过模型参数滑动平均的方式得到更平坦的解，在测试集上具有更好的泛化性能，同时不会增加过多的运算损耗。

+ **There are Many Consistent Explaination of Unlabeled Data: Why You Should Average**
> 通过观察$\pi$-model和Mean Teacher两种半监督模型的训练过程，发现基于一致性的半监督模型在训练尾声阶段依然在使用较大步长探索更优解，因此尝试采用SWA对模型参数进行平均化处理，同时提出fast-SWA，相较于SWA，在同一个学习率变化周期内采样多个模型参数解进行平均，能够更快得到更稳定的解。

+ **Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results**
> 不同于Temporal Ensembling采用预测结果的EMA（指数滑动平均），Mean Teachers采用student模型参数的EMA来得到teacher模型的参数，student模型输出与标签之间的交叉熵作为有监督Loss，student模型输出与teacher模型输出之间的二范数作为无监督Loss。

+ **Temporal Ensembling for Semi-Supervised Learning**
> 提出两种结合有监督Loss和无监督Loss的半监督网络模型，$\pi$-model和Temporal Ensembling。$\pi$-model利用样本两次随机增强和dropout后的输出之间的二范数作为无监督Loss，Temporal Ensembling利用样本当轮输出和之前多轮次输出的EMA（指数滑动平均）之间的二范数作为无监督Loss。

+ **Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks**
> 首先采用抗噪自编码器对网络进行预训练，再利用模型对无标签样本给出最大预测概率的类别作为伪标签，将有标签样本和无标签样本一起输入模型进行fine-tuning，采用交叉熵作为损失函数。

+ **ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring**
> 在MixMatch的基础上，引入两种半监督策略——Distribution Alignment和Augmentation Anchoring。Distribution Alignment采用有标签样本的边际预测分布（所有出现过的有标签样本预测结果的平均）和无标签样本的边际预测分布（对过去128批次无标签样本预测结果的滑动平均）对生成的伪标签进行修正，使得无标签样本的伪标签能够符合有标签样本的标签分布；Augmentation Anchoring采用弱增强样本的伪标签同时作为强增强样本的伪标签，由于直接采用MixMatch中多个增强样本预测结果的平均作为伪标签，在使用强增强时会导致偏差较大，因此首先采用一个弱增强样本的预测结果作为锚定。同时采用基于控制理论的CTAugment生成强增强样本。最后，在半监督的框架下，引入其中一个强增强样本的自监督任务（旋转预测）以提升模型性能。

+ **FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence**
> 一种结合伪标签和一致性原则的半监督方法。对有标签样本来说，采用弱增强样本预测结果和标签之间的交叉熵作为有监督损失；对无标签样本来说，当弱增强样本的预测结果满足一定的置信程度时（即最大预测概率结果高于阈值），其可作为强增强样本的伪标签，采用交叉熵作为无监督损失。

+ **Unsupervised Data Augmentation for Consistency Training**
> 提出在基于一致性的半监督学习中采用有监督学习中有效的数据增强方式，相比简单加噪声的数据增强方式能够更有效提升模型性能。

+ **Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning**
> 对抗训练是一种通过引入模型在扰动下预测标签偏差正则项的方法，其中的扰动能够最大化影响模型的预测标签，而不是采用任意方向的扰动，需要通过梯度上升的方向进行估计，由于需要使用样本的真实标签，因此仅在有监督场景下可以使用。虚拟对抗训练在对抗训练的基础上，采用模型当前的估计值代替标签，因此在有监督和半监督的场景下均能够使用。

+ **Deep Adaptive Image Clustering**
> 一种将表征学习和图像聚类相结合的方法。图像首先被送入卷积网络提取标签特征（label feature），再利用余弦相似性计算两两之间的相似度，当相似度大于或小于特定阈值时可作为相似对和不相似对，其余样本对不使用，据此对卷积网络进行训练；再对阈值参数进行训练，使得阈值上限和阈值下限的差值越来越小，即越来越多的样本能够得到使用，交替重复上述两步训练，直至所有样本都得到使用。当需要给出一张图像的聚类标签时，取标签特征中最大值对应聚类标签即可。

+ **S4L: Self-Supervised Semi-Supervised Learning**
> 在半监督框架下引入自监督，为有标签和无标签样本同时产生旋转角度预测Loss，为有标签样本产生分类Loss。

+ **In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning**
> 提出一种改进伪标签的半监督算法，由于使用错误的伪标签会导致模型向着错误的方向学习发展，因此需要提高伪标签的可靠性。本文采用了预测可信度和不确定性同时筛选非常可靠和非常不可靠的伪标签，其中非常可靠的伪标签即使用普通的交叉熵产生Loss，非常不可靠的伪标签使用改进的负交叉熵产生Loss，因而在保证伪标签可靠性的同时，使用负学习更充分地利用了非常不可靠的伪标签信息，从而使得伪标签方法能够获得和基于一致性的半监督方法相近的性能。

+ **Invariant Information Clustering for Unsupervised Image Classification and Segmentation**
> 提出一种基于互信息最大化的聚类方法，能够直接应用于图像分类和图像分割。首先通过数据增强构造样本对，采用孪生网络结构得出样本对的聚类软标签，最大化样本对软标签之间的互信息用以训练网络，同时可加入辅助聚类分支，使用其他无关类别数据以提升特征提取器能力。

+ **Learning Discrete Representations via Information Maximizing Self-Augmented Training**
> 通过最大化模型输入输出之间的互信息，以及最小化增强样本和原始样本输出之间的交叉熵，实现离散表征的学习，可用于无监督聚类和哈希学习。

+ **Learning Deep Representation by Mutual Information Estimation and Maximization**
> 核心思想是通过最大化同一张图片内局部特征和全局特征之间的互信息实现表征学习。首先将图片分为M×M个子块，每个子块分别提取特征得到局部特征，再将所有子块特征融合得到全局特征，将同一张图片的局部特征和全局特征作为相似样本对，将全局特征和另一张图片的局部特征作为不相似样本对。除此之外，为了使得所提取特征能够含有某些利于下游任务的特性，如正交性等，引入判别器使得所提取特征与先验分布尽可能匹配。

+ **Learning Representations by Maximizing Mutual Information Across Views**
> AMDIM是基于DIM的改进方法，首先使用同一张图片两次不同数据增强后分别用于提取全局特征和局部特征，并且在不同特征尺度上进行正样本对构造，实现多尺度的互信息最大化，同时使用更强大的encoder以及混合特征进一步提升性能。

+ **Barlow Twins: Self-Supervised Learning via Redundancy Reduction**
> 提出一种不同于样本级别的特征级别的对比学习损失函数，将两种不同数据增强后的样本输入孪生网络中得到两组特征，计算两组特征之间的交叉相关系数矩阵，约束使其接近单位阵进行训练，从而实现不同特征之间冗余信息的剔除，同时使得相似样本所提取到的特征接近。

+ **Bootstrap Your Own Latent A New Approach to Self-Supervised Learning**
> 提出一种无需负样本的对比学习方法BYOL，首先对同一张图片进行两次不同的数据增强，将生成的两张增强图片分别输入online网络和target网络（其中target网络参数由online网络动量更新得到），使得online网络提取的特征prediction与target网络提取的特征projection尽可能接近，由于采用非对称网络结构，因此能够在没有负样本的情况下避免模型坍塌的问题。

+ **Exploring Simple Siamese Representation Learning**
> 提出一种简单的孪生网络SimSiam，用于实现无监督表征学习，其整体模型结构和BYOL基本一样（除了去掉了momentum，感觉没有什么区别）。

+ **OpenMatch: Open-set Consistency Regularization for Semi-supervised Learning with Outliers**
> 一种考虑无标签数据中outliers（这里的outliers定义为有标签数据中未出现过的类别）的半监督算法，采用OVA的思想在提取的特征后接上K个二分类器，用于区分inliers和outliers，当K分类器给出概率最大类别对应二分类器判定为outliers时认为该无标签样本为outliers，采用有标签数据、无标签数据熵最小化（使得inliers和outliers的概率差异化增强）和一致性正则项（使得模型输出平滑）对K个二分类器进行训练，采用交叉熵和FixMatch对K分类器进行训练。

+ **CoDiM: Learning with Noisy Labels via Contrastive Semi-Supervised Learning**
> 首先提出CSSL来结合对比学习和半监督学习，利用自监督对比学习进行encoder的训练，再基于encoder结合自监督对比学习、有监督对比学习和半监督学习进行分类器的训练，使得自监督不仅发挥预训练的功能，同时辅助半监督的学习。在CSSL的基础上提出CoDiM进一步解决标签噪声问题，利用自监督对比学习进行encoder的训练，此步并未使用所有的标签信息，再将GMM聚类过程中训练损失低的样本视为标签纯净样本，其余作为标签有噪的无标签样本，最后使用对比学习和半监督学习进行分类器的训练。

+ **DivideMix: Learning with Noisy Labels as Semi-supervised Learning**
> 整体上是一种结合SSL思想用于LNL的方法，首先利用Co-Divide将数据集划分为纯净部分和有噪部分，纯净部分视为有标签，有噪部分视为无标签，再采用SSL方法，并轮流进行这两部分。Co-Divide利用包含两个成分的GMM对样本损失（交叉熵）进行拟合，得到每个样本为纯净样本的概率，根据阈值划分纯净样本和有噪样本，将划分结果传递给另一个模型；采用MixMatch的SSL方法，做了label co-refinement和label co-guessing的改进，label co-refinement针对有标签样本，将原始标签和数据增强后样本估计结果按照纯净概率加权融合，label co-guessing针对无标签样本，将两个模型的结果一起用于伪标签计算。

+ **CoSSL: Co-Learning of Representation and Classifier for Imbalanced Semi-Supervised Learning**
> 提出一种类别不平衡场景下的半监督方法。构建了表征学习、伪标签生成和分类器学习三个模块，表征学习模块的编码器采用EMA方式提供给另外两个模块，分类器学习模块的分类器直接共享给伪标签生成模块，另外分类器学习模块采用TFE策略将有标签数据和无标签数据混合来增加少样本类别的数据多样性，三个模块互相促进使得最终提升在类别不平衡场景下的半监督效果。

+ **Revisiting Consistency Regularization for Semi-Supervised Learning**
> 在一致性正则项的基础上提出将弱增强样本和强增强样本在特征空间中尽可能分离，使得同一类别的样本能够在特征空间内占据更大范围，从而有助于泛化性能提升。同时引入了自监督任务辅助半监督学习。

+ **A Novel Maximum Mean Discrepancy-Based Semi-Supervised Learning Algorithm**
> 提出一种基于MMD的半监督分类算法，其基本思路和self-training一样，通过将伪标签置信度高的无标签样本纳入有标签样本，迭代训练更新模型，区别在于伪标签置信度高的无标签样本的选择基于无标签样本k-means聚类结果和分类器分类结果之间的MMD，将MMD值最小的类对应的无标签样本纳入有标签样本集。

+ **Barely-Supervised Learning: Semi-Supervised Learning with very few labeled images**
> 所提方法针对每个类别有标签样本极少的场景，FixMatch等半监督方法对无标签样本的置信度普遍较低，无法实现有效的半监督学习。因此在无标签样本置信度较高的情况下将弱增强样本的伪标签作为强增强样本的训练目标，在无标签样本置信度较低的情况下采用聚类结果作为训练目标，同时提出两种优化伪标签的方式，第一种没太看懂，第二种是针对不同类别采用不同的置信度阈值，从而对置信度高的无标签样本少的类别降低阈值。