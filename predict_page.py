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

    brands = ('alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
       'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
       'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche', 'renault',
       'saab', 'subaru', 'toyota', 'volkswagen', 'volvo')

    aspirations = ('std', 'turbo')

    brand = st.selectbox('Brand', brands)
    aspiration = st.selectbox('Type', aspirations)

    enginesize = st.slider('engine-size', 60, 320)
    horsepower = st.slider('horsepower', 50, 260)

    ok = st.button('Predict price')
    if ok:
        X = np.array([[brands, aspirations, enginesize, horsepower]])
        X_final = full_pipeline.transform(X)

        price = model.predict(X_final)
        st.subheader(f"The estimated price is ${price[0]:.2f}")