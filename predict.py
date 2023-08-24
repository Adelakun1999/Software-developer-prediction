import streamlit as st
import pickle
import numpy as np



def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor = data['model']
le_country = data['le_country']
le_education = data['le_education']


def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("#### We need some information to predict the salary")

    country = {
        'Australia', 'Austria', 'Brazil', 'Canada', 'France', 'Germany',
        'India', 'Israel', 'Italy', 'Netherlands', 'Poland',
        'Russian Federation', 'Spain', 'Sweden', 'Switzerland', 'Turkey',
        'United Kingdom of Great Britain and Northern Ireland',
        'United States of America'
    }

    education = {
        "Bachelor's degree",
        'Less than a Bachelors', "Master's degree",
        'Post grad'
    }

    country = st.selectbox('Country', country)
    education = st.selectbox('Eduvation Level', education)
    experience = st.slider('Year of experience', 0, 50, 3)

    button = st.button('Calculate Salary')
    if button:
        x = np.array([[country, education, experience]])
        x[:, 0] = le_country.transform(x[:, 0])
        x[:, 1] = le_education.transform(x[:, 1])
        x = x.astype(float)

        salary = regressor.predict(x)
        st.subheader(f"The estimated Salary is ${salary[0]:.2f}")

show_predict_page()