import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

st.header('Welcome to this little Taiwan stock visualization project!')

yf.pdr_override()
start_year = st.selectbox('Please enter the starting year', [
                          i for i in range(2013, 2024)])
end_year = st.selectbox('Please enter the end year:', [
                        i for i in range(2013, 2024)])
view = st.multiselect('Please enter the properties you want to see individually:', [
                      'Open', 'Close', 'Volume', 'High', 'Low'])


if end_year == 2023:
    end_time = datetime.now()
else:
    end_time = f'{end_year}-01-01'

mystock = st.text_input('Please input your stock ID')
st.text('e.g. 精星8183.TWO, just key in 8183')
input_name = f'{mystock}.TW'

try:
    if len(pdr.get_data_yahoo(input_name, start=f'{start_year}-01-01', end=end_time)) == 0:
        input_name = f'{mystock}.TWO'
        df = pdr.get_data_yahoo(
            input_name, start=f'{start_year}-01-01', end=end_time)
except:
    input_name = f'{mystock}.TWO'
    df = pdr.get_data_yahoo(
        input_name, start=f'{start_year}-01-01', end=end_time)

if len(mystock) is 0:
    pass
else:
    st.markdown('Here are the latest five days stock price:')
    st.markdown(f'The input stock name is: {input_name}')
    st.write(df.tail(5))
    # st.line_chart(df[['Close','Volume']])
    df['Volume'] = df['Volume']/1000
    st.line_chart(df[view])
    df = df.reset_index()
    plt.figure(figsize=(10, 8))
    plt.title('Summary')
    ax1 = plt.subplot()
    l1, = ax1.plot(df['Date'], df['Volume'], color='red')
    ax2 = ax1.twinx()
    l2, = ax2.plot(df['Date'], df['Close'], color='blue')
    l3, = ax2.plot(df['Date'], df['High'], color='green')
    l4, = ax2.plot(df['Date'], df['Low'], color='yellow')
    plt.legend([l1, l2, l3, l4], ["Volume", "Close", 'High', 'Low'])
    st.pyplot(plt)
