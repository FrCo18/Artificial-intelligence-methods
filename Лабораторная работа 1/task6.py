import scipy.io as sio
import numpy as np

import svm

dataset = sio.loadmat('dataset2.mat')
X = dataset['X'].astype(np.float32)
y = dataset['y'].astype(np.float32)

C = 1.0
sigma=1.0

gaussian = svm.partial(svm.gaussian_kernel, sigma=sigma)
gaussian.__name__ = svm.gaussian_kernel.__name__
model = svm.svm_train(X, y, C, gaussian)

svm.visualize_boundary(X, y, model, 'Исходные данные')
