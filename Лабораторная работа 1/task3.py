import scipy.io as sio
import numpy as np
import svm

dataset = sio.loadmat('dataset1.mat')
x = dataset['X'].astype(np.float32)
y = dataset['y'].astype(np.float32)

c = 100
model = svm.svm_train(x,y, c, svm.linear_kernel, 0.001, 20)
svm.visualize_boundary_linear(x, y, model, 'Разделяющая граница при C = ' + str(c))
