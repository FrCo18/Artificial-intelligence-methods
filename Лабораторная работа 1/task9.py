import scipy.io as sio
import numpy as np

import svm

dataset = sio.loadmat('dataset3.mat')
X = dataset['X'].astype(np.float32)
y = dataset['y'].astype(np.float32)
Xval = dataset['Xval'].astype(np.float32)
yval = dataset['yval'].astype(np.float32)

for C in[0.01,0.03,0.1,0.3,1,3,10,30]:
    for sigma in[0.01,0.03,0.1,0.3,1,3,10,30]:
        gaussian = svm.partial(svm.gaussian_kernel, sigma=sigma)
        gaussian.__name__ = svm.gaussian_kernel.__name__
        model = svm.svm_train(X, y, C, gaussian)

        ypred = svm.svm_predict(model, Xval)
        error = np.mean(ypred != yval.ravel())

        title1 = f'Обучающая выборка (X={X}, y={y}) (C={C}, sigma={sigma})'
        title2 = f'Тренировочная выборка (Xval={Xval}, yval={yval}) (C={C}, sigma={sigma})'

        svm.visualize_boundary(X, y, model, title1)
        svm.visualize_boundary(Xval, yval, model, title2)

# получение модели с гауссовым ядром, обученной с параметрами С и sigma
# получение результата предсказания для тестовой выборки
# вычисление ошибки предсказания
# если удалось уменьшить ошибку
# запоминаем ошибку, C, sigma и модель








