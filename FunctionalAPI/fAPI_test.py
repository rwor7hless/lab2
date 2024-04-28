import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Загрузка модели
#
loaded_model = load_model('../FunctionalAPI/model_checkpoint.h5')

# Загрузка токенизатора
with open('../FunctionalAPI/tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# Максимальная длина последовательности, используемая при обучении
max_sequence_length = 99


def for_front_Funcstional(review):

    # Максимальная длина последовательности, используемая при обучении
    max_sequence_length = 99

    unlabeled_data = []
    unlabeled_data.append(review)

    # Токенизация неразмеченных данных с использованием загруженного токенизатора и максимальной длины последовательности
    unlabeled_sequences = loaded_tokenizer.texts_to_sequences(unlabeled_data)
    unlabeled_padded_data = pad_sequences(unlabeled_sequences, maxlen=max_sequence_length)

    # Применение загруженной модели к неразмеченным данным
    prediction = loaded_model.predict(unlabeled_padded_data)[0]
    # Вывод результатов
    sentiment = 1 if prediction > 0.5 else 0
    print(f"### Functional ### Прогноз:{sentiment}, Вероятность: {prediction}")
    return sentiment


def for_front_Functional_excel(destination):
    # Максимальная длина последовательности, используемая при обучении
    max_sequence_length = 99

    # Загрузка неразмеченных данных из Excel файла
    unlabeled_data = pd.read_excel(destination, usecols=[0])[
        pd.read_excel(destination, usecols=[0]).columns[0]].tolist()

    # Преобразование значений столбца в строки
    unlabeled_data = [str(data) for data in unlabeled_data]

    # Токенизация неразмеченных данных с использованием загруженного токенизатора и максимальной длины последовательности
    unlabeled_sequences = loaded_tokenizer.texts_to_sequences(unlabeled_data)
    unlabeled_padded_data = pad_sequences(unlabeled_sequences, maxlen=max_sequence_length)

    # Применение загруженной модели к неразмеченным данным
    predictions = loaded_model.predict(unlabeled_padded_data)

    # Создание DataFrame с отзывами и предсказаниями
    df = pd.DataFrame({'Отзыв': unlabeled_data, 'Предсказание': [1 if pred > 0.5 else 0 for pred in predictions]})

    # Записываем предсказания во второй столбец
    df.insert(2, 'predicted_sentiment', [1 if pred > 0.5 else 0 for pred in predictions])

    # Сохранение DataFrame в Excel файл
    df.to_excel(destination, index=False, header=False)

# for_front_Funcstional("мне не нравится этот монитор, не рекомендую к покупке")