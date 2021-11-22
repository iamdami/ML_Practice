import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_CPU_ALLOW_GROWTH'] = 'true'

"""
actions = ['come', 'away', 'spin']

data = np.concatenate([np.load('dataset/seq_come_1637066143.npy'),
                       np.load('dataset/seq_away_1637066143.npy'),
                       np.load('dataset/seq_spin_1637066143.npy')], axis=0)
"""

""""""
actions = ["yes", "hate", "really_hate", "don't"]

data = np.concatenate([np.load('dataset/seq_yes_20211122_000334.npy', allow_pickle=True),
                       np.load('dataset/seq_hate_20211122_000334.npy', allow_pickle=True),
                       np.load('dataset/seq_really_hate_20211122_000334.npy', allow_pickle=True),
                       np.load("dataset/seq_don't_20211122_000334.npy", allow_pickle=True)], axis=0)


print(data)
print('data : ', data.shape)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]

print('x_data : ', x_data.shape)
print('labels', labels.shape)

y_data = to_categorical(labels, num_classes=len(actions))
print('y_data : ', y_data)

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)

fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val_acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()

model = load_model('models/model.h5')
y_pred = model.predict(x_val)

multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
