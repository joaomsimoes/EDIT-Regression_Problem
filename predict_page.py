import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]
full_pipeline = data["pipe"]


def show_predict_page():
    st.title('Auto 1985 Predictor')

    st.write('''We need some information to predict the car price''')

    enginesize = st.slider('engine-size', 60, 320)
    horsepower = st.slider('horsepower', 50, 260)

    curb = st.slider('curb-weight', 1710, 4000)
    citympg = st.slider('city-mpg', 15, 49)

    ok = st.button('Predict price')
    if ok:
        X = np.array([[enginesize, horsepower, curb, citympg]])
        X_final = full_pipeline.transform(X)

        price = model.predict(X_final)
        st.subheader(f"The estimated price is ${price[0]:.2f}")