import string
import joblib
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import nltk
import numpy as np
import pandas as pd


def remove_punctuation(text):
    if isinstance(text, str):
        cleaned_text = ''.join(char for char in text if char not in string.punctuation).lower()
        return cleaned_text
    else:
        return str(text)


nltk.download('punkt')
# Функция загрузки списка слов


def load_word_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        word_list = file.read().splitlines()
    return word_list

# Функция фильтрации текста


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
                negate = False  # сбрасываем флаг отрицания
            else:  # если отрицание неактивно, добавляем слово без "не"
                filtered_text.append(word)
        elif "не" in word:  # если слово содержит "не", но не в списке отрицательных слов
            filtered_text.append(word)
            negate = True  # устанавливаем флаг отрицания

    return ' '.join(filtered_text)

# Функция преобразования отзыва в вектор
def review_to_vector(review_text, word2vec_model):
    if isinstance(review_text, str):  # Проверяем, является ли review_text строкой
        tokens = [word.lower() for word in word_tokenize(review_text) if word.lower() in word2vec_model.wv]
        if tokens:
            return np.mean([word2vec_model.wv[word] for word in tokens], axis=0)
        else:
            return np.zeros(word2vec_model.vector_size)
    else:
        return np.zeros(word2vec_model.vector_size)
# review_from_front = 'Мне очень понравилось'


def W2V_test(review_from_front):
    # Загрузка тестовой выборки
    file_path = review_from_front
    print(file_path)
    # Загрузка моделей
    word2vec_model = Word2Vec.load('../Word2Vec/word2vec_model.bin')
    model = joblib.load('../Word2Vec/mlp_model.joblib')

    # Путь к списку слов
    word_list_path = '../words.txt'
    word_list = load_word_list(word_list_path)

    # Применение фильтрации текста
    file_path = filter_text(file_path, word_list)
    print("после фильтрации:")
    print(file_path)

    # Разделение на токены
    # unlabeled_reviews = file_path.split()

    # Преобразование отзывов в вектора
    unlabeled_vectors = review_to_vector(file_path, word2vec_model)
    # Предсказание тональности
    predictions = model.predict([unlabeled_vectors])[0]

    # Вывод предсказанных тональностей
    print(predictions)
    return 1 if predictions > 0.5 else 0


# Функция загрузки списка слов
def load_word_list_excel(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        word_list = file.read().splitlines()
    return word_list


def W2V_text_excel(destination):
    # Загрузка моделей
    word2vec_model = Word2Vec.load('../word2vec/word2vec_model.bin')
    model = joblib.load('../word2vec/mlp_model.joblib')

    # Путь к списку слов
    word_list_path = "../word2vec/words.txt"
    word_list = load_word_list(word_list_path)

    # Загрузка тестовой выборки
    df = pd.read_excel(destination, usecols=[0, 1, 2, 3, 4], header=None)  # загружаем без заголовков

    # Применение фильтрации текста ко всем отзывам
    filtered_reviews = df[0].apply(lambda x: filter_text(x, word_list))
    # Преобразование отзывов в вектора
    unlabeled_vectors = [review_to_vector(review, word2vec_model) for review in filtered_reviews]

    # Предсказание тональности
    predictions = model.predict(unlabeled_vectors)
    # Преобразование предсказаний в формат 0 и 1
    binary_predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]

    # Объединение временного DataFrame с оригинальным DataFrame
    # запись в 6 столбец
    df['W2W'] = binary_predictions

    # Сохранение результа тов в файл Excel без заголовков
    df.to_excel(destination, index=False, header=False)