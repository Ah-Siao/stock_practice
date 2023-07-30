import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Flatten
from keras import backend as K
import pickle
from custom_trans import transformer_encoder, build_model, lr_scheduler, shift

model_path = './model/'
history_path = './history/'
st.session_state['model'] = 0

input_name = st.text_input('Please enter the stock ID')
st.text("e.g. for 精星 8183.TWO, enter '8183.TWO'")

start_date = str(datetime(2020, 1, 1).date())
end_date = datetime.now().strftime('%Y-%m-%d')
st.text('The model is establish by data from:')
st.text(f'start date:{start_date}')
st.text(f'end data:{end_date}')
st.text('so far, you have model for stock number......')
st.text([i.split('model')[0] for i in os.listdir(model_path)])

while input_name:
    yf.pdr_override()
    stock = pdr.get_data_yahoo(input_name, start=start_date, end=end_date)
    stock_num = input_name.split('.')[0]
    if os.path.exists(f'{model_path}{stock_num}model.pkl'):
        ANSWER = 'NONE'
        while ANSWER == 'NONE':
            ANSWER = st.selectbox(
                'Do you want to rebuild the model?', ['NONE', 'YES', 'NO'])
            if ANSWER == 'YES':
                st.session_state['model'] = 0
            elif ANSWER == 'NO':
                st.session_state['model'] = 1
            elif ANSWER == 'NONE':
                st.text('Please decide whether to build your new model! Thank you!!')

    break

try:
    st.text('stock price of last five days')
    st.write(stock.tail(5))
except:
    pass

target = 'Close'
train_start_date = start_date
train_end_date = '2022-10-31'
test_start_date = '2022-11-01'
training_set = stock.Close[train_start_date:
                           train_end_date].values.reshape(-1, 1)
test_set = stock.Close[test_start_date:].values.reshape(-1, 1)
st.text(f'size of training set: {len(training_set)}')
st.text(f'size of test set: {len(test_set)}')

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
timesteps = 20
x_train = []
y_train = []

for i in range(timesteps, training_set.shape[0]):
    x_train.append(training_set_scaled[i-timesteps:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.LearningRateScheduler(lr_scheduler)
]

input_shape = x_train.shape[1:]
print(input_shape)

model = build_model(
    input_shape,
    head_size=46,  # Embedding size for attention
    num_heads=60,  # Number of attention heads
    ff_dim=55,  # Hidden layer size in feed forward network inside transformer
    num_transformer_blocks=5,
    mlp_units=[256],
    mlp_dropout=0.4,
    dropout=0.14,
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["mean_squared_error"],
)


if st.session_state['model'] == 0:
    my_bar = st.progress(0.0, text='Establishing the model.....')
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=20,
        callbacks=callbacks,
    )
    st.session_state['model'] == 1
    my_bar.progress(1.0, text='Successfully build the model!')
    filename = f'model/{stock_num}model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    df = pd.DataFrame(history.history)
    df.to_csv(f'history/{stock_num}history.csv')

elif st.session_state['model'] == 1:
    st.text('Let us use our old model!')


# load the model and history:
df = pd.read_csv(f'history/{stock_num}history.csv')

with open(f"model/{stock_num}model.pkl", 'rb') as f:
    model = pickle.load(f)

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(3, 1, 1)
plt.title('loss')
plt.plot(df[['loss', 'val_loss']])
plt.legend(['loss', 'val_loss'])
fig.add_subplot(3, 1, 2)
plt.plot(df[['mean_squared_error', 'val_mean_squared_error']])
plt.legend(['mean_squared_error', 'val_mean_squared_error'])
plt.title('mean sqaure error')
fig.add_subplot(3, 1, 3)
plt.plot(df['lr'])
plt.title('learning rate')
plt.legend('learning rate')
st.pyplot(fig)


dataset_total = pd.concat(
    (stock[target][:train_end_date], stock[target][test_start_date:]), axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set) - timesteps:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.fit_transform(inputs)

X_test = []
for i in range(timesteps, test_set.shape[0] + timesteps):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

st.text(f'R2 score:{r2_score(test_set,predicted_stock_price)}')

plt.figure(figsize=(10, 8))
plt.title(f'{input_name}')
ax1 = plt.subplot()
l1, = ax1.plot(predicted_stock_price, color='red')
l2, = ax1.plot(test_set, color='blue')
plt.legend([l1, l2], ["Predicted Price", "Real Price"])
st.pyplot(plt)

# Predict tomorrow's stock price
last_few_days = stock[-(timesteps):].Close.values
last_few_days = last_few_days.reshape(-1, 1)
last_few_days = sc.fit_transform(last_few_days)
last_few_days = last_few_days[np.newaxis, :]
last_few_days.shape
y_pred = model.predict(last_few_days)
y_pred = sc.inverse_transform(y_pred)
st.text(f'the next stock price should be: {np.around(y_pred.ravel()[0], 2)}')
