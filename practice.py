import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# 1. Load your assets locally inside your function
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('colab_titanic_ann_model.h5')
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.bind(file) if hasattr(pickle, 'bind') else pickle.load(file)
    with open('onehot_encoder.pkl', 'rb') as file:
        onehot_encoder = pickle.load(file)
    return model, scaler, onehot_encoder

model, scaler, onehot_encoder = load_assets()

# ... (Your Streamlit UI input widgets here: pclass, sex, age, sibsp, parch, fare, embarked)

st.title('Passenger Survival Chance in the Titanic Journey')
st.write('Provide input details to check survival probability.')

pclass=st.slider('Enter Passenger Class', 1,3,value=3)
sex=st.selectbox('Enter the sex',['male','female'])
sibsp=st.number_input('Enter the number of Siblings/Spouse',1,10,value=1)
parch=st.number_input('Enter the number of Parents/Children',1,10,value=1)
fare=st.number_input('Enter the Fare paid',0.00,500.00,value=50.00)
embarked=st.selectbox('Enter the Embarked Station', ['Cherobogh', 'Southampton','Queenstown'])

if st.button("Predict Survival"):
# Step A: Scale your numerical features using your loaded scaler 
# (Matches ['SibSp', 'Parch', 'Fare'])
    raw_numerical = pd.DataFrame([{'SibSp': sibsp, 'Parch': parch, 'Fare': fare}])
    scaled_numerical = scaler.transform(raw_numerical)[0] 
    
# Step B: Correctly use One-Hot Encoder on categorical inputs together
# We pass them as a 2D array/DataFrame matching how the OHE expects it
    raw_categorical = pd.DataFrame([{'Sex': sex, 'Embarked': embarked}])
    
# .toarray() ensures it converts from a sparse matrix to a usable numpy array
    encoded_categorical = onehot_encoder.transform(raw_categorical).toarray()[0]
    
# Step C: Combine scaled numericals and encoded categoricals horizontally
# This creates the exact sequence array your ANN model expects!
    final_features = np.hstack([scaled_numerical, encoded_categorical]).reshape(1, -1)
    
# 2. Make the ANN Prediction
    prediction = model.predict(final_features)[0][0]
    
# 3. Display Results
    st.subheader("Result")
    if prediction > 0.5:
        st.success(f"Passenger is likely to Survive! 🎉 (Probability: {prediction:.2%})")
    else:
        st.error(f"Passenger is unlikely to Survive. ❌ (Probability: {prediction:.2%})")



   
