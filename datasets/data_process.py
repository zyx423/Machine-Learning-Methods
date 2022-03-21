import scipy.io as scio
import numpy as np
import os



class LoadData():
    def __init__(self, dataset):
        self.dataset = dataset
        self.path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), './.'))
        print(self.path)

    def mat(self):

        path = self.path + '/{}.mat'.format(self.dataset)
        data = scio.loadmat(path)

        labels = data['Y']
        if labels.shape[0] == 1:
            labels = np.reshape(labels, (labels.shape[1], 1))

        features = data['X']
        # change the data into n×d
        if features.shape[1] == labels.shape[0]:
            features = features.T

        return features, labels

# if __name__ == '__main__':
#     loaddata = LoadData('ATT')
#     features, labels = loaddata.mat()
#     print(type(features))
#     print(labels)