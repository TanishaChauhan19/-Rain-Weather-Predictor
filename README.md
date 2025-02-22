# -Rain-Weather-Predictor
This project is a Django web application that provides weather forecasts based on the user's input city. It leverages weather data from the OpenWeatherMap API and incorporates machine learning techniques to predict future weather conditions, such as temperature, humidity, and rain chances. The web application integrates classification and regression models for predicting rainfall and future temperature/humidity.

The project utilizes various Python libraries like scikit-learn, Pandas, and Imbalanced-learn for data handling, model training, and prediction. Additionally, it uses SMOTE (Synthetic Minority Over-sampling Technique) to balance imbalanced datasets.

Key Features:
1. Weather Forecast: Users can enter a city name to retrieve real-time weather information, including current temperature, wind speed, humidity, and pressure.
The weather data is fetched from the OpenWeatherMap API and displayed on the web page.

2. Rain Prediction: A RandomForestClassifier model is used to predict whether it will rain tomorrow based on historical weather data. The model is trained on a dataset that includes features like temperature, wind speed, and humidity.
The model also uses class weights to handle class imbalance in predicting rain.

3. Future Temperature and Humidity Prediction: RandomForestRegressor models are used to predict future temperature and humidity values. The models are trained using historical data, and the application forecasts weather for the next 5 hours.
The future temperature and humidity values are predicted iteratively and displayed to the user.

4. Interactive Forecasts:Users receive hourly forecasts for the next 5 hours (temperature and humidity).
The system converts wind direction data into compass points (e.g., North, South, East) to display the wind's direction in a human-readable format.

5. Time Zone Support: The application considers the local time zone of the user (in this case, Asia/Kolkata) to generate accurate time predictions for future forecasts.

## Screenshot

Here’s a screenshot of the app:

  ![Image Description](https://github.com/TanishaChauhan19/-Rain-Weather-Predictor/blob/main/weatherApp%20image.jpg?raw=true)

  ## Installation

To install this app, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/TanishaChauhan19/-Rain-Weather-Predictor.git

2. Navigate to the project directory:

       cd -Rain-Weather-Predictor
    
4. Set Up a Virtual Environment
  
       python -m venv myenv
4.Activate the virtual environment:
    
       
    myenv\Scripts\activate
5.Install Libraries
  for example:Django
    
    pip install Django
6.Setting Up the Database
    
     
     # Ensure you're inside the project directory where `manage.py` is located (inside `weatherProject`)
     
     cd weatherProject          
     python manage.py migrate 
     
7.Running the Development Server


    cd weatherProject  # Navigate to the `weatherProject` folder if not already inside
    python manage.py runserver




