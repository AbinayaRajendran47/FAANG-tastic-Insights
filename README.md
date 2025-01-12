# FAANG-tastic-Insights

Project Overview
The goal of this project is to predict stock prices for FAANG companies (Facebook, Amazon, Apple, Netflix, Google) using historical stock data. The project follows a structured workflow encompassing data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment. The MLflow library was utilized to track experiments, compare models, register the best-performing model, and deploy it to a Streamlit application for real-time predictions.
FAANG/
│
├── app.py                  # Streamlit application for deployment
├── faang.ipynb             # Jupyter notebook for experimentation and model development
├── FAANG - FAANG.csv       # Historical stock data for FAANG companies
├── readme.md               # Project README file
│
├── mlartifacts/            # Directory for storing MLflow artifacts
│
├── mlruns/                 # Directory for MLflow experiment tracking
│
└── myenv/                  # Virtual environment or dependency management folder
