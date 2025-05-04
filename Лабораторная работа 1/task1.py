import scipy.io as sio
import numpy as np
import svm

dataset = sio.loadmat('dataset1.mat')
x = dataset['X'].astype(np.float32)
y = dataset['y'].astype(np.float32)

svm.visualize_boundary_linear(x, y, None, 'Исходные данные')
