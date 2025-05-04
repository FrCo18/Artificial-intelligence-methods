import numpy as np
import scipy.io as sio
from sklearn import svm
from collections import OrderedDict

from twisted.python.util import println

from process_email import process_email
from process_email import email_features
from process_email import get_dictionary

with open('email.txt', 'r') as file:
    email = file.read().replace('\n', '')

email = process_email(email)
println(email)

emailFeatures = email_features(email)
println('Длина вектора признаков: {0}'.format(len(emailFeatures)))
println('Количество ненулевых элементов: {0}'.format(sum(emailFeatures > 0)))