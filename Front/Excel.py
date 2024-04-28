import pandas as pd


def count_voices_excel(destination):
    pos_count = 0
    neg_count = 0
    df = pd.read_excel(destination, usecols=[0, 1, 2, 3, 4, 5], header=None)
    df.columns = ['Отзывы', 'Нейрон', 'Functional', 'Sequential', 'FastText', 'Word2Vec']
    for index, row in df.iterrows():
        voices_1 = sum(row.iloc[1:])
        voices_0 = len(row.iloc[1:]) - voices_1
        if voices_1 > voices_0:
            final_pred = 1
            pos_count += 1
            df.at[index, 'Результат'] = final_pred
        else:
            final_pred = 0
            neg_count += 1
            df.at[index, 'Результат'] = final_pred
    df.at[1, 'Всего положительных'] = pos_count
    df.at[1, 'Всего негативных'] = neg_count
    df.to_excel(destination, index=False)