import pandas as pd 
import numpy as np 
import datetime
import pickle 
import streamlit as st 

# loading in the model to predict on the data 
pickle_in = open("models/rand_forest_regressor.pkl", "rb") 
rand_forest_regressor = pickle.load(pickle_in) 

pickle_in = open("models/XGB_model.pkl", "rb")  
XGB_model = pickle.load(pickle_in) 

pickle_in = open("models/lgb_model.pkl", "rb") 
lgb_model = pickle.load(pickle_in) 

pickle_in = open("models/reg_model.pkl", "rb") 
reg_model = pickle.load(pickle_in) 

def date_convert(date_time):
    format = '%Y-%m-%d %H:%M:%S'
    datetime_str = datetime.datetime.strptime(date_time, format)
 
    return datetime_str

# defining the function which will make the prediction using 
# the data which the user inputs 
def prediction( 
	VendorID, 
	passenger_count, 
	trip_distance, 
	RatecodeID,
	store_and_fwd_flag,
	PULocationID,
	DOLocationID,
	payment_type,
	Pickup_date
	): 
	# Pickup realted date parser
	Pickup_date = date_convert(Pickup_date)
	hour = Pickup_date.hour
	day = Pickup_date.day
	month = Pickup_date.month
	year = Pickup_date.year
	day_of_week = Pickup_date.weekday()

	rand_forest_regressor_result = rand_forest_regressor.predict( 
		[
			[ 
				VendorID, passenger_count, trip_distance, RatecodeID, store_and_fwd_flag, 
				PULocationID, DOLocationID, payment_type, hour, month, year, day_of_week
		    ]
		]
	)

	XGB_model_result = XGB_model.predict( 
		[
			[ 
				VendorID, passenger_count, trip_distance, RatecodeID, store_and_fwd_flag, 
				PULocationID, DOLocationID, payment_type, hour, month, year, day_of_week
		    ]
		]
	) 

	lgb_model_result = lgb_model.predict( 
		[
			[ 
				VendorID, passenger_count, trip_distance, RatecodeID, store_and_fwd_flag, 
				PULocationID, DOLocationID, payment_type, hour, month, year, day_of_week
		    ]
		]
	) 

	reg_model_result = reg_model.predict( 
		[
			[ 
				VendorID, passenger_count, trip_distance, RatecodeID, store_and_fwd_flag, 
				PULocationID, DOLocationID, payment_type, hour, month, year, day_of_week
		    ]
		]
	) 
	return rand_forest_regressor_result, XGB_model_result, lgb_model_result, reg_model_result
	

# this is the main function in which we define our webpage 
def main(): 
	# giving the webpage a title 
	st.title("Yellow Taxi Fare Prediction") 
	
	# here we define some of the front end elements of the web page like 
	# the font and background color, the padding and the text to be displayed 
	html_temp = """ 
	<div style ="background-color:yellow;padding:13px"> 
	<h1 style ="color:black;text-align:center;">Streamlit Yellow Taxi Fare Prediction ML App </h1> 
	</div> 
	"""
	
	# this line allows us to display the front end aspects we have 
	# defined in the above code 
	st.markdown(html_temp, unsafe_allow_html = True) 
	
	# the following lines create text boxes in which the user can enter 
	# the data required to make the prediction 
	Pickup_date = st.text_input("Pickup date", "2017-12-01 00:12:00")
	VendorID = int(st.text_input("Vendor ID", "1"))
	passenger_count = int(st.text_input("Passenger count", "1"))
	trip_distance = float(st.text_input("Trip distance", "4.20"))
	RatecodeID = int(st.text_input("Ratecode ID", "1"))
	store_and_fwd_flag = int(st.text_input("store and forward flag", "0"))
	PULocationID = int(st.text_input("Pickup Location ID", "82"))
	DOLocationID = int(st.text_input("Destination Location ID", "252"))
	payment_type = int(st.text_input("Payment type", "2"))
	result ="" 
	
	# the below line ensures that when the button called 'Predict' is clicked, 
	# the prediction function defined above is called to make the prediction 
	# and store it in the variable result 
	if st.button("Predict"): 
		rand_forest_regressor_result, XGB_model_result, lgb_model_result, reg_model_result = prediction(
			VendorID, passenger_count, trip_distance, RatecodeID, store_and_fwd_flag, PULocationID, DOLocationID, payment_type, Pickup_date
		)
		result = f"""Random Forest Regressor Model Prediction {rand_forest_regressor_result} 
		             XGBoost Model Prediction {XGB_model_result} 
					 LightGBM Model Prediction {lgb_model_result} 
					 Linear Regression Model Prediction {reg_model_result}"""
	st.success(result) 
	
if __name__=='__main__': 
	main() 
