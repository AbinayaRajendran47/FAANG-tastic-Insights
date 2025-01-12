# import streamlit as st
# import pandas as pd
# import mlflow.pyfunc
# import numpy as np
# import matplotlib.pyplot as plt

# # Set the MLflow tracking URI
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# # Load the registered model
# model_name = "rf_model"
# model_version = 5  # Change the version as needed
# model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# # title
# st.title("Stock Price Prediction")

# st.markdown("""
# This application predicts the closing price of selected stocks based on user-provided input features.
# """)

# # User Input Section
# st.header("Input Stock Details")

# # Input fields for user to enter stock data
# open_price = st.number_input("Open Price", value=100.0)
# high_price = st.number_input("High Price", value=105.0)
# low_price = st.number_input("Low Price", value=95.0)
# volume = st.number_input("Volume", value=1_000_000.0)
# adj_close = st.number_input("Adjusted Close", value=100.0)

# # Day of the week selection
# day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
# dow_encoded = {
#     "Monday": [1, 0, 0, 0, 0],
#     "Tuesday": [0, 1, 0, 0, 0],
#     "Wednesday": [0, 0, 1, 0, 0],
#     "Thursday": [0, 0, 0, 1, 0],
#     "Friday": [0, 0, 0, 0, 1],
# }[day_of_week]

# # Month selection
# month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
# month_encoded = {
#     "January": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "February": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "March": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "April": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     "May": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     "June": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     "July": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     "August": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     "September": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     "October": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#     "November": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#     "December": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
# }[month]

# # Year input (default to 2024)
# year_encoded = st.number_input("Year", value=2024)

# # Ticker selection (One-hot encoding for simplicity)
# company = st.selectbox("Select Company Stock", ["AMAZON", "APPLE", "FACEBOOK", "GOOGL", "NETFLIX"])
# stock_amazon = 1 if company == "AMAZON" else 0
# stock_apple = 1 if company == "APPLE" else 0
# stock_facebook = 1 if company == "FACEBOOK" else 0
# stock_google = 1 if company == "GOOGL" else 0
# stock_netflix = 1 if company == "NETFLIX" else 0

# # Prepare input DataFrame for prediction


# user_input = pd.DataFrame({
#     'Open': [open_price],
#     'High': [high_price],
#     'Low': [low_price],
#     'Volume': [volume],
#     'Adj Close': [adj_close],
#     'Stock_Amazon': [stock_amazon],
#     'Stock_Apple': [stock_apple],
#     'Stock_Facebook': [stock_facebook],
#     'Stock_Google': [stock_google],
#     'Stock_Netflix': [stock_netflix],
#     'day_of_week_Monday': [dow_encoded[0]],
#     'day_of_week_Tuesday': [dow_encoded[1]],
#     'day_of_week_Wednesday': [dow_encoded[2]],
#     'day_of_week_Thursday': [dow_encoded[3]],
#     'day_of_week_Friday': [dow_encoded[4]],
#     **dict(zip([f'month_{i+1}' for i in range(12)], month_encoded)),  # Dynamically map month encodings
#     'year_encoded': [year_encoded]  # Use the input year
# })

# # Correct feature order
# expected_features = [
#     'Open', 'High', 'Low', 'Volume', 'Adj Close', 'Stock_Amazon', 'Stock_Apple',
#     'Stock_Facebook', 'Stock_Google', 'Stock_Netflix',
#     'day_of_week_Monday', 'day_of_week_Tuesday', 'day_of_week_Wednesday',
#     'day_of_week_Thursday', 'day_of_week_Friday',
#     'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
#     'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'year_encoded'
# ]

# # Reorder user_input DataFrame
# user_input = user_input[expected_features]

# # Display user input as a table
# st.subheader("Input Data Overview")
# st.dataframe(user_input)

# # Prediction button
# if st.button("Predict Closing Price"):
#     try:
#         # Make prediction
#         prediction_log = model.predict(user_input)
#         prediction = np.exp(prediction_log)  # Back-transform if the model output is in log scale

#         # Display the predicted stock price
#         st.header("Predicted Stock Price")
#         st.write(f"The predicted stock closing price is: ${prediction[0]:.2f}")

#         # Visualization: Plot the input features and predicted price
#         st.subheader("Prediction Visualization")
#         fig, ax1 = plt.subplots(figsize=(10, 6))

#         # Plot the numerical inputs as bars
#         ax1.bar(['Open', 'High', 'Low', 'Volume'],
#                 [open_price, high_price, low_price, volume],
#                 color='skyblue', label='Input Features')
#         ax1.set_ylabel('Values')
#         ax1.set_xlabel('Features')
#         ax1.tick_params(axis='y')
#         ax1.legend(loc='upper left')

#         # Overlay the predicted price as a line
#         ax2 = ax1.twinx()
#         ax2.plot(['Prediction'], [prediction[0]],
#                  color='orange', marker='o', label='Prediction')
#         ax2.set_ylabel('Price in $')
#         ax2.legend(loc='upper right')

#         # Set plot title
#         plt.title('Input Features and Predicted Stock Price')
#         st.pyplot(fig)

#     except Exception as e:
#         st.error(f"An error occurred while predicting: {e}")
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the saved best model (best_model.pkl)
with open('best_RandomForest.pkl', 'rb') as f:
    best_model = pickle.load(f)

# title
st.title("Stock Price Prediction")

st.markdown("""
This application predicts the closing price of selected stocks based on user-provided input features.
""")

# User Input Section
st.header("Input Stock Details")

# Input fields for user to enter stock data
open_price = st.number_input("Open Price", value=100.0)
high_price = st.number_input("High Price", value=105.0)
low_price = st.number_input("Low Price", value=95.0)
volume = st.number_input("Volume", value=1_000_000.0)
adj_close = st.number_input("Adjusted Close", value=100.0)

# Day of the week selection
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
dow_encoded = {
    "Monday": [1, 0, 0, 0, 0],
    "Tuesday": [0, 1, 0, 0, 0],
    "Wednesday": [0, 0, 1, 0, 0],
    "Thursday": [0, 0, 0, 1, 0],
    "Friday": [0, 0, 0, 0, 1],
}[day_of_week]

# Month selection
month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
month_encoded = {
    "January": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "February": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "March": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "April": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "May": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "June": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "July": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "August": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "September": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "October": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "November": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "December": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}[month]

# Year input (default to 2024)
year_encoded = st.number_input("Year", value=2024)

# Ticker selection (One-hot encoding for simplicity)
company = st.selectbox("Select Company Stock", ["AMAZON", "APPLE", "FACEBOOK", "GOOGL", "NETFLIX"])
stock_amazon = 1 if company == "AMAZON" else 0
stock_apple = 1 if company == "APPLE" else 0
stock_facebook = 1 if company == "FACEBOOK" else 0
stock_google = 1 if company == "GOOGL" else 0
stock_netflix = 1 if company == "NETFLIX" else 0

# Prepare input DataFrame for prediction
user_input = pd.DataFrame({
    'Open': [open_price],
    'High': [high_price],
    'Low': [low_price],
    'Volume': [volume],
    'Adj Close': [adj_close],
    'Stock_Amazon': [stock_amazon],
    'Stock_Apple': [stock_apple],
    'Stock_Facebook': [stock_facebook],
    'Stock_Google': [stock_google],
    'Stock_Netflix': [stock_netflix],
    'day_of_week_Monday': [dow_encoded[0]],
    'day_of_week_Tuesday': [dow_encoded[1]],
    'day_of_week_Wednesday': [dow_encoded[2]],
    'day_of_week_Thursday': [dow_encoded[3]],
    'day_of_week_Friday': [dow_encoded[4]],
    **dict(zip([f'month_{i+1}' for i in range(12)], month_encoded)),
    'year_encoded': [year_encoded]
})

# Correct feature order (ensure same order as in the trained model)
expected_features = [
    'Open', 'High', 'Low', 'Volume', 'Adj Close', 'Stock_Amazon', 'Stock_Apple',
    'Stock_Facebook', 'Stock_Google', 'Stock_Netflix',
    'day_of_week_Monday', 'day_of_week_Tuesday', 'day_of_week_Wednesday',
    'day_of_week_Thursday', 'day_of_week_Friday',
    'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
    'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'year_encoded'
]

# Reorder user_input DataFrame
user_input = user_input[expected_features]

# Display user input as a table
st.subheader("Input Data Overview")
st.dataframe(user_input)

# Prediction button
if st.button("Predict Closing Price"):
    try:
        # Make prediction
        prediction_log = best_model.predict(user_input)
        prediction = np.exp(prediction_log)  # Back-transform if the model output is in log scale

        # Display the predicted stock price
        st.header("Predicted Stock Price")
        st.write(f"The predicted stock closing price is: ${prediction[0]:.2f}")

        # Visualization: Plot the input features and predicted price
        st.subheader("Prediction Visualization")
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the numerical inputs as bars
        ax1.bar(['Open', 'High', 'Low', 'Volume'],
                [open_price, high_price, low_price, volume],
                color='skyblue', label='Input Features')
        ax1.set_ylabel('Values')
        ax1.set_xlabel('Features')
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')

        # Overlay the predicted price as a line
        ax2 = ax1.twinx()
        ax2.plot(['Prediction'], [prediction[0]],
                 color='orange', marker='o', label='Prediction')
        ax2.set_ylabel('Price in $')
        ax2.legend(loc='upper right')

        # Set plot title
        plt.title('Input Features and Predicted Stock Price')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while predicting: {e}")
