# Classifying Airline Passenger Satisfaction
This project uses a Kaggle data and fits classification models that balances recall and precision metrics with an emphasis on model interpretability. The dataset includes trip specific data, passenger specific data, and survey data on various components of the air travel experience. The use case for these results is to empower airlines with information regarding passenger's expectations for the air travel experience. With these insights, airlines can triangulate what causes dissastisfaction, optimize the categories causing dissatisfaction, and predict whether a customer is dissatisfied without a customer participating in a post flight survey.c
**Data Source**: https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction

**Tools Used:**
* Postgres
* SQL
* Python 3
* SciKit Learn
* Streamlit

## Conclusions
The best model to meet this project's goal is logistic regression. The top 5 features that airline pasengers care about most are as follows:

1. Inflight Wifi Service
2. Type of Travel (Business/Personal)
3. Check In Service
4. Type of Customer (Loyal/Disloyal)
5. Cleanliness

## Outline of Files
1. The ipynb files contain exploratory data analyatics and techniques used for model tuning and selection.
2. The pickle file contains pickles of the final trained logistic regression model, the coeffiecients, and the scaler.
3. The CSV files contains the Kaggle data set (divided into train and test) and the cleaned dataframe.
4. The airline_data.py files contains code for the interactive visualization on Streamlit.
5. The pdf contains the final presentation.

