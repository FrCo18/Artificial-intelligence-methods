import numpy as np
import scipy.io as sio
from sklearn import svm
from collections import OrderedDict

from process_email import process_email
from process_email import email_features
from process_email import get_dictionary

with open('email.txt', 'r') as file:
    email = file.read().replace('\n', '')

print(email)