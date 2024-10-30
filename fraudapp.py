import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import InputLayer

# Streamlit app layout
st.title("Fraud Transaction Detection")

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.input_layer = Input(shape=(6,), name='input_layer')
        self.dense1 = Dense(64, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.output_layer(x)

model = MyModel()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
input_layer = InputLayer(input_shape=(6,), name='input_layer')



# User input fields
TX_AMOUNT = st.number_input("Transaction Amount")
TERMINAL_ID = st.text_input("Terminal ID")
CUSTOMER_ID = st.text_input("Customer ID")
TX_DATETIME = st.date_input("Transaction Date")
TX_DATETIME_hour = st.number_input("Transaction Hour")

# When 'Detect Fraud' button is clicked
if st.button("Detect Fraud"):

    is_high_amount = TX_AMOUNT > 220
    is_fraud_terminal = False
    is_high_spend_customer = False

    # Creating input data
    input_data = np.array([[TX_AMOUNT, is_high_amount, is_fraud_terminal, is_high_spend_customer, TX_DATETIME.day, TX_DATETIME_hour]])

    # Prediction
    prediction = model.predict(input_data)
    result = "Fraudulent" if prediction[0][0] > 0.5 else "Legitimate"
    st.write(f"Transaction is {result}.")
