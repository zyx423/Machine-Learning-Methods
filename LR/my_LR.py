

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn import datasets


# 手写数字样本。
# 共有1797个样本，每个样本有64的元素，对应到一个8x8像素点组成的矩阵，每一个值是其灰度值， target值是0-9，适用于分类任务。
iris_dataset = datasets.load_digits()
features, labels = iris_dataset.data, iris_dataset.target

# 划分训练集和测试集。我这里假设：70%数据做训练集，30%数据做测试集
scale = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=scale, random_state=0)

# 初始化LR模型
# 使用L2正则化
LR = LogisticRegression()
LR.fit(X_train, Y_train)
LR.predict(X_test)
score = LR.score(X_test, Y_test)

print(score)
