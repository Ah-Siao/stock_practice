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


def transformer_encoder(inputs,head_size,num_heads,ff_dim,dropout=0):
    x=layers.LayerNormalization(epsilon=1e-6)(inputs)
    x=layers.MultiHeadAttention(
        key_dim=head_size,num_heads=num_heads,dropout=dropout
    )(x,x)
    x=layers.Dropout(dropout)(x)
    res=x+inputs
    x=layers.LayerNormalization(epsilon=1e-6)(res)
    x=layers.Conv1D(filters=ff_dim,kernel_size=1,activation='relu')(x)
    x=layers.Dropout(dropout)(x)
    x=layers.Conv1D(filters=inputs.shape[-1],kernel_size=1)(x)
    return x+res

def build_model(input_shape, head_size,num_heads,ff_dim, num_transformer_blocks, mlp_units,dropout=0,mlp_dropout=0):
    inputs=keras.Input(shape=input_shape)
    x=inputs
    for _ in range(num_transformer_blocks):
        x=transformer_encoder(x,head_size,num_heads,ff_dim,dropout)
    x=layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    for dim in mlp_units:
        x=layers.Dense(dim,activation='elu')(x)
        x=layers.Dropout(mlp_dropout)(x)
    outputs=layers.Dense(1,activation='linear')(x)
    return keras.Model(inputs,outputs)

def lr_scheduler(epoch, lr, warmup_epochs=30, decay_epochs=100, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr


def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e


input_name=st.text_input('Please enter the stock ID')
st.text("e.g. for 精星 8183.TWO, enter '8183.TWO'")

start_date = str(datetime(2020,1,1).date())
end_date = datetime.now().strftime('%Y-%m-%d')
st.text('The model is establish by data from:')
st.text(f'start date:{start_date}')
st.text(f'end data:{end_date}')


while input_name:
    yf.pdr_override()
    stock=pdr.get_data_yahoo(input_name, start=start_date, end=end_date)
    break

try:
    st.write(stock.tail(5))
except:
    pass


target='Close'
train_start_date=start_date
train_end_date='2022-10-31'
test_start_date='2022-11-01'
training_set=stock.Close[train_start_date:train_end_date].values.reshape(-1,1)
test_set=stock.Close[test_start_date:].values.reshape(-1,1)
st.text(f'size of training set: {len(training_set)}')
st.text(f'size of test set: {len(test_set)}')

sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)
timesteps=20
x_train=[]
y_train=[]

for i in range(timesteps,training_set.shape[0]):
    x_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

idx=np.random.permutation(len(x_train))
x_train=x_train[idx]
y_train=y_train[idx]

callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.LearningRateScheduler(lr_scheduler)
            ]

input_shape = x_train.shape[1:]
print(input_shape)

model = build_model(
    input_shape,
    head_size=46, # Embedding size for attention
    num_heads=60, # Number of attention heads
    ff_dim=55, # Hidden layer size in feed forward network inside transformer
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

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=20,
    callbacks=callbacks,
)


df=pd.DataFrame(history.history)

fig=plt.figure(figsize=(10,10))
fig.add_subplot(3,1,1)
plt.title('loss')
plt.plot(df[['loss','val_loss']])
plt.legend(['loss','val_loss'])
fig.add_subplot(3,1,2)
plt.plot(df[['mean_squared_error','val_mean_squared_error']])
plt.legend(['mean_squared_error','val_mean_squared_error'])
plt.title('mean sqaure error')
fig.add_subplot(3,1,3)
plt.plot(df['lr'])
plt.title('learning rate')
plt.legend('learning rate')
st.pyplot(fig)



dataset_total = pd.concat((stock[target][:train_end_date],stock[target][test_start_date:]),axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set) - timesteps:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.fit_transform(inputs)

X_test = []
for i in range(timesteps,test_set.shape[0] + timesteps):
    X_test.append(inputs[i-timesteps:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

st.text(f'R2 score:{r2_score(test_set,predicted_stock_price)}')

plt.figure(figsize=(10,8))
plt.title(f'{input_name}')
ax1=plt.subplot()
l1,=ax1.plot(predicted_stock_price,color='red')
l2,=ax1.plot(test_set,color='blue')
plt.legend([l1,l2], ["Predicted Price", "Real Price"])
st.pyplot(plt)

#Predict tomorrow's stock price
last_few_days=stock[-(timesteps):].Close.values
last_few_days=last_few_days.reshape(-1,1)
last_few_days=sc.fit_transform(last_few_days)
last_few_days=last_few_days[np.newaxis,:]
last_few_days.shape
y_pred=model.predict(last_few_days)
y_pred = sc.inverse_transform(y_pred)

st.text(f'the next stock price should be: {round(y_pred.ravel()[0],2)}')













