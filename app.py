import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="EV Charging Demand Predictor", layout="wide")
st.title("🚗 EV Charging Demand Prediction")
st.markdown("Predict the energy demand (kWh) based on time, weather, and station occupancy.")

# 2. Mock Data Generation (Replace with your CSV)
@st.cache_data
def load_data():
    # In a real project, use: pd.read_csv('ev_data.csv')
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="H")
    data = pd.DataFrame({
        'timestamp': dates,
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'temperature': np.random.uniform(10, 35, size=1000),
        'is_holiday': np.random.choice([0, 1], size=1000, p=[0.9, 0.1]),
        'charging_demand': np.random.uniform(50, 500, size=1000)
    })
    # Add some noise/patterns
    data['charging_demand'] += data['hour'] * 5 + data['temperature'] * 2
    return data

df = load_data()

# 3. Sidebar - Feature Inputs
st.sidebar.header("User Input Parameters")
input_hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
input_temp = st.sidebar.slider("Temperature (°C)", 0, 45, 25)
input_day = st.sidebar.selectbox("Day of Week", options=range(7), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
input_holiday = st.sidebar.radio("Is it a Holiday?", [0, 1])

# 4. Model Training (XGBoost)
# Preparing Features
X = df[['hour', 'day_of_week', 'temperature', 'is_holiday']]
y = df['charging_demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 5. Prediction
user_input = np.array([[input_hour, input_day, input_temp, input_holiday]])
prediction = model.predict(user_input)[0]

# 6. Display Results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted Demand")
    st.metric(label="Estimated Load", value=f"{prediction:.2f} kWh")
    
    # Feature Importance Plot
    importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    fig_imp = px.bar(importance, x='feature', y='importance', title="Feature Importance")
    st.plotly_chart(fig_imp, use_container_width=True)

with col2:
    st.subheader("Historical Trends")
    fig_trend = px.line(df.tail(100), x='timestamp', y='charging_demand', title="Recent Charging Demand")
    st.plotly_chart(fig_trend, use_container_width=True)

st.success("The model predicts demand based on historical usage patterns and environmental factors.")
