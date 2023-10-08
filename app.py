import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


lr_model = "models/LR_model.pkl"
knn_model = "models/KNN_model.pkl"
svm_model = "models/SVM_model.pkl"

with open(lr_model, 'rb') as file:
    LR_model = pickle.load(file)

with open(knn_model, 'rb') as file:
    KNN_model = pickle.load(file)

with open(svm_model, 'rb') as file:
    SVM_model = pickle.load(file)

st.title("Parkinson's Disease Prediction Using Machine Learning")

st.header('Fill the form and press the predict button to see the result',
          divider='rainbow')

# test_value = [[ 5.29249395e-01, -1.03309592e-01,  1.11583374e+00,
#         -5.23716022e-01, -6.82179904e-01, -4.29193964e-01,
#         -4.66158519e-01, -4.28206106e-01, -6.17874540e-01,
#         -6.05206208e-01, -6.44461440e-01, -5.48107644e-01,
#         -5.58034900e-01, -6.44471776e-01, -5.39840575e-01,
#          7.18861885e-01, -1.49332475e+00,  1.18499869e+00,
#         -3.23304568e-01, -3.76742214e-01,  3.78931110e-01,
#         -3.93143882e-01]]


# Initial values
initial_values = {
    'MDVP:Fo(Hz)': 119.99200,
    'MDVP:Fhi(Hz)': 157.30200,
    'MDVP:Flo(Hz)': 74.99700,
    'MDVP:Jitter(%)': 0.00784,
    'MDVP:Jitter(Abs)': 0.00007,
    'MDVP:RAP': 0.00370,
    'MDVP:PPQ': 0.00554,
    'Jitter:DDP': 0.01109,
    'MDVP:Shimmer': 0.04374,
    'MDVP:Shimmer(dB)': 0.42600,
    'Shimmer:APQ3': 0.02182,
    'Shimmer:APQ5': 0.03130,
    'MDVP:APQ': 0.02971,
    'Shimmer:DDA': 0.06545,
    'NHR': 0.02211,
    'HNR': 21.03300,
    'RPDE': 0.414783,
    'DFA': 0.815285,
    'spread1': -4.813031,
    'spread2': 0.266482,
    'D2': 2.301442,
    'PPE': 0.284654
}


# for key, value in initial_values.items():
#     st.text_input(key, value=value)

for i, (key, value) in enumerate(initial_values.items()):
    initial_values[key] = st.text_input(key, value=value, key=i)

# print(list(enumerate(initial_values.items())))

if st.button('Predict', type="primary"):
    # Get the values from the text input fields
    # values = [float(st.text_input(key, value=value)) for key, value in initial_values.items()]
    values = [float(value) for value in initial_values.values()]

    # Convert the list of values to a numpy array and store it in 'test_value'
    test_value = np.array(values)
    sc = StandardScaler()

    # Print 'test_value' to the console
    st.write("Logistic Regression", str(
        LR_model.predict(sc.fit_transform([test_value]))))
    st.write("KNN", str(KNN_model.predict(sc.fit_transform([test_value]))))
    st.write("SVM", str(SVM_model.predict(sc.fit_transform([test_value]))))
    st.write("Provided Data: ", test_value)

dataset = pd.read_csv("dataset/parkinsons.data")

st.subheader('Dataset Sample:', divider='rainbow')
st.write(dataset.head())
