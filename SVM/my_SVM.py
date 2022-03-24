import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn import datasets


# 手写数字样本。
# 共有1797个样本，每个样本有64的元素，对应到一个8x8像素点组成的矩阵，每一个值是其灰度值， target值是0-9，适用于分类任务。
iris_dataset = datasets.load_digits()
features, labels = iris_dataset.data, iris_dataset.target


# 划分训练集和测试集。我这里假设：70%数据做训练集，30%数据做测试集
scale = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=scale, random_state=0)

# SVM既可以用来分类，就是SVC；
# 又可以用来预测，或者成为回归，就是SVR。
# sklearn中的svm模块中也集成了SVR类。
clf = svm.SVC()
clf.fit(X_train, Y_train)
res_svc = clf.score(X_test, Y_test)
print('使用SVM做分类的结果是：{:.2f}%'.format(res_svc*100))


# 这里使用SVR做预测，随便从测试集中选出一个样本预测其标签。
clr = svm.SVR()
clr.fit(X_train, Y_train)
test_sample = np.reshape(X_test[0], (1, -1))
res_svr = clr.predict(test_sample)
print('这个样本使用SVM做预测的结果是：{}'.format(int(np.round(res_svr))))
