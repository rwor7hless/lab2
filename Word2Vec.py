import string

import openpyxl
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib


file_path = '../reviews.xlsx'

# Импорт данных из Excel файла
df = pd.read_excel(file_path)



def load_word_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        word_list = file.read().splitlines()
    return word_list


word_list_path = '../words.txt'
word_list = load_word_list(word_list_path)

def remove_punctuation(text):
    # Удаляем знаки препинания из текста
    cleaned_text = ''.join(char for char in text if char not in string.punctuation).lower()
    return cleaned_text
def filter_text(text, word_list):
    text = remove_punctuation(text)
    words = text.split()
    filtered_text = []
    negate = False  # флаг, который будет указывать на текущий отрицательный контекст

    for word in words:
        if word == "не":  # если слово "не" обнаружено, меняем флаг отрицания
            negate = True
        elif any(root in word for root in word_list):  # проверяем наличие корня из списка слов в текущем слове
            if negate:  # если флаг отрицания установлен
                filtered_text.append("не " + word)
                negate = False
            else:  # если отрицание неактивно, добавляем слово без "не"
                filtered_text.append(word)
        elif "не" in word:  # если слово содержит "не", но не в списке отрицательных слов
            filtered_text.append(word)
    return ' '.join(filtered_text)
def load_data(file_path, word_list):
    workbook = openpyxl.load_workbook(file_path, read_only=True)
    sheet = workbook.active

    texts = []
    labels = []

    for row in sheet.iter_rows(min_row=1, max_row=1000, values_only=True):
        text = row[0] if row[0] is not None else ""
        filtered_text = filter_text(text, word_list)
        texts.append(filtered_text)

        label = row[1] if row[1] is not None else 0
        labels.append(int(label))

    print(texts, labels)
    return texts, labels


# Загрузка данных
texts, labels = load_data(file_path, word_list)


# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=42)
#
# Обучение модели Word2Vec на обучающей выборке
sentences = [word_tokenize(review.lower()) for review in X_train]
word2vec_model = Word2Vec(sentences, vector_size=25, window=2, min_count=1, epochs=250, batch_words=16)

# Преобразование отзывов в векторы
def review_to_vector(review):
    tokens = [word.lower() for word in word_tokenize(review) if word.lower() in word2vec_model.wv]
    if tokens:
        return np.mean([word2vec_model.wv[word] for word in tokens], axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)
X_train_vectors = [review_to_vector(review) for review in X_train]
X_test_vectors = [review_to_vector(review) for review in X_test]

# Преобразование меток классов в массив NumPy
y_train = np.array(y_train)
y_test = np.array(y_test)
# Создание и обучение нейронной сети
model = MLPClassifier(hidden_layer_sizes=(32,16,8,1), solver= "adam", activation='relu', max_iter=250,
                      shuffle=False, batch_size=16, learning_rate= "adaptive", learning_rate_init=0.0001,)
model.fit(X_train_vectors, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test_vectors)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность на обучающей выборке: {accuracy * 100:.2f}%')
# сохранение моделей
joblib.dump(model, 'mlp_model.joblib')
model_path = "word2vec_model.bin"
word2vec_model.save(model_path)