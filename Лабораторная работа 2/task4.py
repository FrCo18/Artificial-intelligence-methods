import numpy as np
import scipy.io as sio
from sklearn import svm
from collections import OrderedDict

from twisted.python.util import println

from process_email import process_email
from process_email import email_features
from process_email import get_dictionary

# Загружаем обучающие данные из train.mat
data = sio.loadmat('train.mat')
X_train = data['X']
y_train = data['y'].flatten()

# обучение
print("Тренировка SVM-классификатора с линейным ядром...")
clf = svm.SVC(C=0.1, kernel='linear', tol=1e-3)
model = clf.fit(X_train, y_train)
predictions = model.predict(X_train)

# оценка точности
accuracy = np.mean(predictions == y_train)
print(f"Точность на обучающей выборке: {accuracy * 100:.3f}%")


