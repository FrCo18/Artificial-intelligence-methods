# Импортируем необходимые модули
from process_email import process_email, email_features, get_dictionary
from sklearn import svm
import scipy.io
import numpy as np

# Загружаем модель, подготовленную ранее
data = scipy.io.loadmat('train.mat')
X_train = data['X']
y_train = data['y'].flatten()

# Тренируем модель SVM
clf = svm.SVC(C=0.1, kernel='linear', tol=1e-3)
model = clf.fit(X_train, y_train)


# Функция для обработки письма и вывода результата
def check_email(file_path):
    try:
        # Открываем письмо из файла
        with open(file_path, 'r') as f:
            email_text = f.read()

        # Предобрабатываем письмо
        processed_email = process_email(email_text)

        # Генерируем вектор признаков
        feature_vector = email_features(processed_email)

        # Прогнозируем класс письма
        prediction = model.predict(feature_vector.reshape(1, -1))[0]

        return prediction
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None


# Путь к файлу хорошего письма
good_email_path = 'good_email.txt'

# Путь к файлу спам-письма
spam_email_path = 'spam.txt'

# Проверяем оба письма
good_prediction = check_email(good_email_path)
spam_prediction = check_email(spam_email_path)

# Анализ результатов
if good_prediction is not None:
    if good_prediction == 1:
        print("Ваше хорошее письмо ошибочно помечено как СПАМ.")
    else:
        print("Ваше хорошее письмо не является спамом.")

if spam_prediction is not None:
    if spam_prediction == 1:
        print("Ваше спам-письмо успешно идентифицировано как СПАМ.")
    else:
        print("Ошибка! Ваш спам неправильно определен как НЕСПАМ.")
