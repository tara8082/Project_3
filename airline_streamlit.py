import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import math

st.title("Predicting Airline Passenger Satisfaction")

# Load data frame
data = pd.read_csv('df_clean.csv', index_col = 0)

#st.dataframe(data) displays dataframe

# Loading models and coefficients
coefficients = pkl.load(open('/Users/Tara8082/GIT/ProjectGIT/Project_3/streamlit_logistic_coef3.pkl', 'rb'))
log_model = pkl.load(open('/Users/Tara8082/GIT/ProjectGIT/Project_3/streamlit_logistic_model3.pkl', 'rb'))
ss = pkl.load(open('/Users/Tara8082/GIT/ProjectGIT/Project_3/standard_scaler3.pkl', 'rb'))


st.write("This is an application for predicting airline passenger satisfaction with a trained model from user inputs. Let's try it out!")

# Check box for displaying the data
check_data = st.checkbox("See the sample data.")
if check_data:
    st.write(data.head(50))

st.write('Let us find out whether a passenger is "dissatisfied/neutral" or "satisfied" when we choose parameters.')

# User inputs for features
departure_delay = st.number_input('Departure Delay In Minutes', value = 0)
inflight_wifi_service = st.number_input('Inflight Wifi Service', value= 0)
customer_type = st.multiselect('Customer Type',['Loyal', 'Disloyal'], default = 'Loyal')
type_of_travel = st.multiselect('Type Of Travel',['Personal', 'Business'], default = 'Business')
seat_comfort = st.number_input('Seat Comfort', value = 0)
airline_class = st.multiselect('Class',['Eco', 'Eco Plus', 'Business'], default = 'Business')
inflight_service = st.number_input('Inflight Service', value = 3)
ease_of_online_booking = st.number_input('Ease Of Online Booking', value = 0)
gate_location = st.number_input('Gate Location', value = 3)
online_boarding = st.number_input('Online Boarding',value = 3)
inflight_entertainment = st.number_input('Inflight Entertainment', value = 3)
food_and_drink = st.number_input('Food And Drink',value = 3)
cleanliness = st.number_input('Cleanliness', value = 3)
gender = st.multiselect('Gender', ['Female', 'Male'], default = 'Female')
age = st.number_input('Age', value = 35)
flight_distance = st.number_input('Flight Distance', value = 500)
dep_arr_time_convenience = st.number_input('Departure/Arrival Time Convenience', value = 3)
onboard_service = st.number_input('Onboard Service', value = 3)
leg_room_service = st.number_input('Legroom Service', value = 3)
baggage_handling = st.number_input('Baggage Handling', value = 3)
checkin_service = st.number_input('Checkin Service', value = 3)

gender_male = 0
gender_female = 0
customer_type_disloyal = 0
customer_type_loyal = 0
type_of_travel_Business = 0
type_of_travel_Personal = 0
class_business = 0
class_eco = 0
class_eco_plus =0

if gender == 'Female':
    gender_female = 1
else:
    gender_male = 1

if customer_type == 'Disloyal':
    customer_type_disloyal = 1
else:
    customer_type_loyal = 1

if type_of_travel == 'Business':
    type_of_travel_Business = 1
else:
    type_of_travel_Personal = 1

if airline_class == 'Eco':
    class_eco = 1
elif airline_class == 'Eco Plus':
    class_eco_plus = 1
else:
    class_business = 1




# Setting up input data dictionary:
input_data = {'Age': [age], 'Flight Distance' : [flight_distance], 'Inflight Wifi Service': [inflight_wifi_service], 
                            'Dep Arr Time Convenience': [dep_arr_time_convenience], 'Ease Of Online Booking': [ease_of_online_booking],
                            'Gate Location': [gate_location], 'Food And Drink':[food_and_drink], 'Online Boarding': [online_boarding],
                            'Seat Comfort' : [seat_comfort], 'Inflight Entertainment': [inflight_entertainment], 'Onboard Service': [onboard_service],
                            'Leg Room Service': [leg_room_service], 'Baggage Handling': [baggage_handling], 'Checkin Service': [checkin_service],
                            'Inflight Service': [inflight_service], 'Cleanliness':[cleanliness], 'Departure Delay In Minutes': [departure_delay],
                            'Gender_Female': [gender_female], 'Gender_Male': [gender_male], 'Customer Type_Disloyal': [customer_type_disloyal], 'Customer Type_Loyal': [customer_type_loyal],
                            'Type of Travel_Business travel': [type_of_travel_Business], 'Type of Travel_Personal Travel': [type_of_travel_Personal], 'Class_Business': [class_business],
                            'Class_Eco': [class_eco], 'Class_Eco Plus': [class_eco_plus]}
                            
input_data = pd.DataFrame.from_dict(input_data, orient = 'columns')

# Preprocessing input data
#categoricals = ['Gender', 'Customer Type', 'Type Of Travel', 'Class']

# non_cat = ['Flight Distance', 'Inflight Wifi Service',
#        'Dep Arr Time Convenience', 'Ease Of Online Booking',
#        'Gate Location', 'Food And Drink', 'Online Boarding', 'Seat Comfort',
#        'Inflight Entertainment', 'Onboard Service', 'Leg Room Service',
#        'Baggage Handling', 'Checkin Service', 'Inflight Service',
#        'Cleanliness', 'Departure Delay In Minutes']

# ohe = OneHotEncoder(sparse=False)
# cat_matrix_input = ohe.fit_transform(input_data.loc[:, categoricals])
 
# input_ohe = pd.DataFrame(cat_matrix_input,
#                            columns=ohe.get_feature_names(categoricals),
#                            index=input_data.index)

# st.write(input_ohe)

# # Combine continuous and categorical input data
# input_data_combo = pd.concat([input_data.drop(columns = categoricals), input_ohe], axis=1)


# st.write('input data', input_data.shape)
# st.write('input_ohe', input_ohe.shape)
# st.write('input_data.drop', input_data[non_cat].shape)
# st.write('input_data_combo', input_data_combo.shape)
# st.write(input_data_combo)

# Scale input data
input_data_preprocessed = ss.fit_transform(input_data)

def prediction(input_data_preprocessed):
    return log_model.predict(input_data_preprocessed)

predict_button = st.button('Predict')

if predict_button:
    result = prediction(input_data_preprocessed)[0]
    if result == 1:
        st.write('The passenger is "Dissatisfied/Neutral".')
    else:
        st.write('The passenger is "Satisfied".')

st.markdown('Created by **Tara Ziegler**')