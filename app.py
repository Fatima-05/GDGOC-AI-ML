import streamlit as st
import joblib 
import numpy as np

# Load the trained model
model = joblib.load('titanic_rf_model.pk1')

# Set up the Streamlit app
st.title('Titanic Survival Predictor')

# Create input fields for user to enter passenger details
age = st.number_input('Age', 0,100,25)
fare=st.number_input('Fare', 0.0, 500.0, 50.0)
sex= st.selectbox('Sex', ['male', 'female'])
pclass = st.radio('Passenger Class', [1, 2, 3])

# Convert categorical input to numerical format
sex_encoded= 0 if sex == 'male' else 1

input_data = np.array([[age, fare, sex_encoded, pclass]])

# Make predictions

prediction = model.predict(input_data)
probability = model.predict_proba(input_data)[0, 1]

if st.button('Predict Survival'):
    if prediction[0]==1:
        st.success("Passenger will survive")
    else:
        st.error("Passenger will not survive")
    st.subheader("Survival Probability")
    st.write(f'{probability:.2%}')
