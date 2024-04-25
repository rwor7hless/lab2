import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from data.Neuron.superAI import *
from tensorflow.keras.optimizers import Adam
import pickle
np.random.seed(42)
# Отзывы и метки
reviews = all_revs
labels = np.array(labels)

# Ваш готовый датасет слов для обучения
word_dataset = dataset_words

# Создаем токенизатор с правильным num_words
tokenizer = Tokenizer(num_words=len(word_dataset), oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
max_sequence_length = max([len(seq) for seq in sequences])
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Создаем модель с регуляризацией L2
input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=len(word_dataset), output_dim=16, input_length=max_sequence_length)(input_layer)
spatial_dropout = SpatialDropout1D(0.2)(embedding_layer)
lstm_layer = LSTM(8, kernel_regularizer=l2(0.1), recurrent_regularizer=l2(0.1))(spatial_dropout)
output_layer = Dense(1, activation='sigmoid')(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callback для сохранения весов при достижении точности 97%
checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='accuracy', save_best_only=True, mode='max', verbose=1)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Обучение модели
model.fit(data, labels, epochs=60, batch_size=8, callbacks=[checkpoint])


# Применение модели к новым отзывам
new_reviews = check_revs

# Преобразуем новые отзывы в последовательности
new_sequences = tokenizer.texts_to_sequences(new_reviews)
new_data = pad_sequences(new_sequences, maxlen=max_sequence_length)

predictions = model.predict(new_data)
count = 0
for i in range(len(predictions)):
    print(f"Отзыв: '{i+1}' - Прогноз: {'Положительный' if predictions[i] > 0.5 else 'Отрицательный'}")
    if (i <= 2 and predictions[i] < 0.5) or (i==3 and predictions[i] > 0.5) or ((i == 4 or i == 5) and predictions[i] < 0.5) or ((i > 5 and i <= 17) and predictions[i] > 0.5) or ((i > 17 and i <=48) and predictions[i] < 0.5) or ((i > 48 and i < 99) and predictions[i] > 0.5):
        count+=1


# Сохранение весов Embedding слоя
embedding_weights = model.layers[1].get_weights()[0]
np.savetxt('embedding_weights.txt', embedding_weights, fmt='%.18e', delimiter=' ')

with open('embedding_weights.txt', 'w', encoding='utf-8') as file:
    for word, index in tokenizer.word_index.items():
        if index < len(word_dataset):
            embedding_vector = embedding_weights[index]
            file.write(f"{word}\n")
print(count)