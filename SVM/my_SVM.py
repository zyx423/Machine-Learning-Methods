import os
import sys
import numpy as np

# 调用数据集中的LoadData类
path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../.'))
sys.path.append(path + '\datasets')
from data_process import LoadData

# 直接用sklearn中的SVM
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score


# 使用ATT，也可以换成别的数据集。
# 这个数据集有400个样本，每个样本有1024个特征，一共40个类
dataset = 'ATT'
loaddata = LoadData(dataset)
features, labels = loaddata.mat()


# 划分训练集和测试集
# 我这里假设：70%数据做训练集，30%数据做测试集
scale = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=scale, random_state=0)

# SVM既可以用来分类，就是SVC；
# 又可以用来预测，或者成为回归，就是SVR。
# sklearn中的svm模块中也集成了SVR类。
clf = svm.SVC()
clf.fit(X_train, Y_train)
# 评估标准用的f1_score, 可以看作是模型精确率和召回率的一种调和平均。
Pred_Y = clf.predict(X_test)
res_svc = f1_score(Pred_Y, Y_test, average='weighted')
print('使用SVM做分类的结果是：{:.2f}%'.format(res_svc*100))


# 这里使用SVR做预测，随便从测试集中选出一个样本预测其标签。
clr = svm.SVR()
clr.fit(X_train, Y_train)
test_sample = np.reshape(X_test[0], (1, -1))
res_svr = clr.predict(test_sample)
print('使用SVM做预测的结果是：{}'.format(int(np.round(res_svr))))
