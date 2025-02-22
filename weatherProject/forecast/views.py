from django.shortcuts import render
from  django.http import HttpResponse

# Create your views here.
import requests  #this library helps us to fetch data from api
import pandas as pd #for handling and analysing data
import numpy as np #for numerical operations
from sklearn.model_selection import train_test_split #to split data into training and test sets
from sklearn.preprocessing import LabelEncoder #to convert categorical data into numericals values
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #models for classification and regression tasks.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #to measure the accuracy of the predictions.
from datetime import datetime,timedelta #to handle data and time
import pytz
import os
from imblearn.over_sampling import SMOTE  # For oversampling (if you decide to use SMOTE)
from sklearn.utils.class_weight import compute_class_weight

API_KEY='Your access key '
base_url='https://api.openweathermap.org/data/2.5/' #base url for making api requests

def validate_city(city):
    # Check if the city is not empty and contains only letters and spaces
    if not city or not city.replace(' ', '').isalpha():
        return False
    return True

def get_current_weather(city):
    url = f"{base_url}weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful (status code 2xx)
        data = response.json()

        # Check if city was found
        if data.get("cod") != 200:
           raise ValueError("City not found.")

        return {
            'city': data['name'],
            'current_temp': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity']),
            'description': data['weather'][0]['description'],
            'country': data['sys']['country'],
            'wind_gust_dir': data['wind']['deg'],
            'pressure': data['main']['pressure'],
            'Wind_Gust_Speed': data['wind']['speed'],
            'clouds': data['clouds']['all'],
            'visibility': data['visibility'],
        }
    except (requests.exceptions.RequestException, ValueError) as e:
        # Catch both request errors and invalid city errors
        print(f"Error: {e}")
        return None

def read_historical_data(filename):
  df=pd.read_csv(filename) #load csv file into dataframe
  df=df.dropna() #remove rows with missing values
  df=df.drop_duplicates()
  return df

def prepare_data(data):
  le = LabelEncoder() #create a LabelEncoder instance
  data['WindGustDir']=le.fit_transform(data['WindGustDir'])
  data['RainTomorrow']=le.fit_transform(data['RainTomorrow'])

  #define the feature variable and target variables
  x=data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']] #feature variables
  y=data['RainTomorrow'] #target variable
  return x,y,le #return feature variable,target variable,and the label encoder.


def train_rain_model(x, y):
    # Split the dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Option 1: Use Class Weights to address class imbalance
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    # Alternatively, if you want to use manual class weights based on your data
    # class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
    # model = RandomForestClassifier(n_estimators=100, class_weight={0: class_weights[0], 1: class_weights[1]}, random_state=42)

    model.fit(x_train, y_train)  # Train the model

    y_pred = model.predict(x_test)  # Predict on the test set

    # Metrics Calculation
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

  
    # Get predicted probabilities for the positive class (rain)
    y_probs = model.predict_proba(x_test)[:, 1]  # Probability of rain

# Adjust the threshold (set to 0.25)
    y_pred_adjusted = (y_probs > 0.25).astype(int)  # Classify as rain if probability > 0.25

# Calculate precision, recall, and F1 score for the adjusted predictions
    precision = precision_score(y_test, y_pred_adjusted)
    recall = recall_score(y_test, y_pred_adjusted)
    f1 = f1_score(y_test, y_pred_adjusted)

# Print the results
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')


    return model


def prepare_regression_data(data,feature):
  x,y=[],[]#intialize list for features and target values

  for i in range(len(data)-1):
    x.append(data[feature].iloc[i])
    y.append(data[feature].iloc[i+1])
  x=np.array(x).reshape(-1,1)
  y=np.array(y)
  return x, y

def train_regression_model(x,y):
  model=RandomForestRegressor(n_estimators=100,random_state=42)
  model.fit(x,y)
  return model

def predict_future(model,current_value):
  predictions=[current_value]

  for i in range(5):
    next_value=model.predict(np.array([[predictions[-1]]]))
    predictions.append(next_value[0])

  return predictions[1:]

#weather analysis

def weather_view(request):
    if request.method=='POST':
        city=request.POST.get('city')
        if not validate_city(city):
            return render(request, 'weather.html', {'error': "Invalid city name. Please try again."})
        current_weather = get_current_weather(city)
        if current_weather is None:
            return render(request, 'weather.html', {'error': "City not found or invalid. Please try again."})

        # load historical data

        path= r'C:\Users\gaura\Desktop\p\Weather\weatherProject\weather.csv'
        historical_data = read_historical_data(path)

        # prepare and train the rain prediction model

        x, y, le = prepare_data(historical_data)

        rain_model = train_rain_model(x, y)
        # map wind direction to campass points
        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
                          ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
                          ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
                          ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
                          ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
                          ("NNW", 326.25, 348.75)
                          ]
        compass_direction = next((point for point, start, end in compass_points if start <= wind_deg < end),None)
        if compass_direction is None:
            return render(request, 'weather.html', {'error': "Wind degree invalid. Please try again."})
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        current_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather['Wind_Gust_Speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp'],
        }

        current_df = pd.DataFrame([current_data])
        # rain prediction

        rain_prediction = rain_model.predict(current_df)[0]  # 0 extract single predicted value

        # prepare regression model for temperature and humidity
        x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')

        x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

        temp_model = train_regression_model(x_temp, y_temp)

        hum_model = train_regression_model(x_hum, y_hum)

        # predict future temperature and humidity

        future_temp = predict_future(temp_model, current_weather['temp_min'])

        future_humidity = predict_future(hum_model, current_weather['humidity'])

        # prepare time for future predictions
        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        #store each value separately

        time1,time2,time3,time4,time5=future_times
        temp1,temp2,temp3,temp4,temp5=future_temp
        hum1,hum2,hum3,hum4,hum5=future_humidity

        #pass data to template
        context = {
            'location':city,
            'current_temp': current_weather['current_temp'],
            'MinTemp': current_weather['temp_min'],
             'MaxTemp': current_weather['temp_max'],
             'feels_like':current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'],
            'city': current_weather['city'],
            'country': current_weather['country'],

            'time':datetime.now(),
            'date':datetime.now().strftime("%m/%d/%Y"),
            'wind':current_weather['Wind_Gust_Speed'],
            'pressure':current_weather['pressure'],
            'temp':current_weather['temp_min'],
            'visibility':current_weather['visibility'],

            'time1':time1,
            'time2':time2,
            'time3':time3,
            'time4':time4,
            'time5':time5,

            'temp1':f"{round(temp1,1)}",
            'temp2':f"{round(temp2,1)}",
            'temp3':f"{round(temp3,1)}",
            'temp4':f"{round(temp4,1)}",
            'temp5':f"{round(temp5,1)}",

            'hum1':f"{round(hum1,1)}",
            'hum2':f"{round(hum2,1)}",
            'hum3':f"{round(hum3,1)}",
            'hum4':f"{round(hum4,1)}",
            'hum5':f"{round(hum5,1)}",

        }

        return render(request, 'weather.html', context)

  #  return render(request, 'weather.html')
    return render(request, 'weather.html', {'error': "Enter a city name to get the weather forecast."})




c
