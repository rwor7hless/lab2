import fasttext
import pandas as pd

# Загрузка модели с main файла
model = fasttext.load_model('../FastText/model.bin')

def FastText(review_from_front):
    # Загрузка модели с main файла
    review = review_from_front
    # Проверка работы модели на неразмеченных отзывах
    prediction = model.predict(review)
    print(f'Отзыв: {review} \n Прогноз: {prediction[0][0]}, Вероятность: {prediction[1][0]:.4f}\n')
    if prediction[0][0] == "__label__1":
        return 1
    else:
        return 0

def FastText_excel(destination):
    # Загрузка данных из Excel файла
    df = pd.read_excel(destination, usecols=[0, 1, 2, 3], header=None)

    # Добавление нового заголовка
    # df.columns = ['Reviews', 'Predictions_Neuron1', 'Predictions_Functional', 'Predictions_Sequential']

    # Загрузка модели с main файла
    model = fasttext.load_model('../FastText/model.bin')

    # Предсказания FastText для каждого отзыва
    predictions = []
    for review in df.iloc[:, 0]:
        prediction = model.predict(str(review))
        predictions.append(int(prediction[0][0][-1]))

    # Добавление предсказаний от новой нейронной сети в новый столбец DataFrame
    df['FT'] = predictions

    # Сохранение DataFrame в Excel файл
    df.to_excel(destination, index=False, header=False)