import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

st.title("Predicting Airline Passenger Satisfaction")

data = pd.read_csv('df_clean.csv', index_col = 0)
st.dataframe(data)

with open('/Users/Tara8082/GIT/ProjectGIT/Project_3/streamlit_app_model.pkl', 'rb') as f:
    model = pkl.load(f)

st.write(
'''
## Make predictions with the trained model from user input
'''
)

departure_delay = st.number_input('Departure Delay in Minutes', value=30)  # this is input option
inflight_wifi_service = st.number_input('Inflight Wifi Service', value=3)

input_data = data({'departure_delay': ['Departure Delay In Minutes'], 'inflight_wifi_service': ['Inflight Wifi Service']})
pred = model.predict(input_data)#[0]
st.write(
'Predicted Satisfaction:' + pred
)


#streamlit loads models
#model needs to take in
#Xgboost may only take array with a single row data frame
#Create a few values
#some can be hard defined
#st.input text   for every feature that you want to include
#just need to make the row for the input


#'Gender', 'Customer Type', 'Age', 'Type Of Travel', 'Class', 
#'Flight Distance', 'Inflight Wifi Service', 'Dep Arr Time Convenience', 'Ease Of Online Booking', 
#'Gate Location', 'Food And Drink','Online Boarding', 'Seat Comfort', 'Inflight Entertainment'
#Onboard Service', 'Leg Room Service', 'Baggage Handling','Checkin Service', 'Inflight Service', 'Cleanliness','Departure Delay In Minutes'