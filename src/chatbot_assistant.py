import streamlit as st
import numpy as np
import random
import re
import time
import string
import datetime as dt
import joblib
import holidays
import redis
import openrouteservice
import configparser
import os
from sklearn.pipeline import Pipeline
from opencage.geocoder import OpenCageGeocode
from geopy.distance import geodesic
from datetime import datetime, timedelta
from typing import Tuple

def stream_simulation(assistant_response):
    message_placeholder = st.empty()
    full_response = ""
    for chunk in assistant_response.split(): #split the text from assistant per space
        full_response += chunk + " "
        time.sleep(0.05)
        message_placeholder.markdown(full_response + "▌")
    return (message_placeholder, full_response)

def remove_punctuation(input_string):
    # Create a translation table that maps each punctuation character to None (i.e., removes them)
    translator = str.maketrans('', '', string.punctuation)
    # Use the translate method to remove punctuation
    cleaned_string = input_string.translate(translator)
    return cleaned_string

def eta_duration(pickup_time:str, predicted_duration:float):
    hours, minutes = map(int, pickup_time.split(':'))
    int_minutes_to_add = int(predicted_duration)

    original_time = dt.datetime(1, 1, 1, hours, minutes)
    new_time = original_time + dt.timedelta(minutes=int_minutes_to_add)

    # Format the result as "hour:minute"
    result_time_str = new_time.strftime("%H:%M")
    return result_time_str

def retrieve_trip_metadata(pickup_location:str, dropoff_location:str)-> Tuple[dict, dict, dict, dict]:
    key = '8d9425286cb64486aae1d5000472b211'
    geocoder = OpenCageGeocode(key)

    query_from = f'{pickup_location}'.lower() + u', New York, USA'.lower()
    query_to = f'{dropoff_location}'.lower() + u', New York, USA'.lower()

    go_from:dict = geocoder.geocode(query_from)[0]
    go_to:dict = geocoder.geocode(query_to)[0]
    neighborhood_from:dict = geocoder.geocode(f"{go_from['components']['neighbourhood']}".lower() + u', New York, USA'.lower())[0]
    neighborhood_to:dict = geocoder.geocode(f"{go_to['components']['neighbourhood']}".lower() + u', New York, USA'.lower())[0]
    return go_from, go_to, neighborhood_from, neighborhood_to

def compute_trip_and_centroid_distance(destination_metadata:Tuple[dict, dict, dict, dict]) -> Tuple[float, float]:
    go_from, go_to, neighborhood_from, neighborhood_to = destination_metadata
    client = openrouteservice.Client(key='5b3ce3597851110001cf624841589dcc471d49fbb2b9a6f24ea4e804')
    distance_coords:tuple = (
        (go_from['geometry']['lng'], go_from['geometry']['lat']), #pikcup (lng, lat)
        (go_to['geometry']['lng'], go_to['geometry']['lat']) #dropoff (lng, lat)
    )
    # Calculate directions from A to B
    routes = client.directions(distance_coords, profile="driving-car")
    trip_distance:float = routes['routes'][0]['summary']['distance'] / 1000

    centoid_distance:float = geodesic(
        (neighborhood_from['geometry']['lat'], neighborhood_from['geometry']['lng']),
        (neighborhood_to['geometry']['lat'], neighborhood_to['geometry']['lng'])
    ).kilometers
    return trip_distance, centoid_distance

def retrieve_day_time_metadata(day:str, time:str)-> Tuple:
    day_mapping:dict = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }
    us_holidays = holidays.country_holidays('US', years=range(datetime.now().date().year, (datetime.now() + timedelta(days=1*365)).year))
    hol_dts = []
    for date, name in sorted(us_holidays.items()):
        hol_dts.append(date)
    if day == "today":
        dt = datetime.now().date()
    elif day == "tomorrow":
        dt = datetime.now().date() + timedelta(days=1)
    elif day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        current_dt = datetime.now().date()
        days_ahead = day_mapping[day] - current_dt.weekday() + 7
        dt = current_dt + timedelta(days=days_ahead)
    else:
        clean_dt = re.sub(u'[./,-]', '/', day).split('/')
        dt = datetime(year=datetime.now().year, month=int(clean_dt[0]), day=int(clean_dt[1])).date()

    (year_dt, month_dt, day_dt) = dt.year, dt.month, dt.day
    #--------------------------------------------------------------------------------------
    pickup_weekday = dt.weekday() + 1
    pickup_holiday = int(dt in hol_dts)
    pickup_weekend = int(pickup_weekday in [6,7])
    #--------------------------------------------------------------------------------------
    hour_tm, minute_tm = ( int(time.split(":")[0]), int(time.split(":")[1]) )
    pickup_hour = hour_tm
    pickup_daytime = 1 if (hour_tm in range(7,11)) or (hour_tm in range(16,20)) else 2 if hour_tm in [20,21,22,23,0,1,2,3,4,5,6] else 3
    pickup_quarter =  (hour_tm * 60 + minute_tm) // 15 + 1
    month_start = datetime(year=year_dt, month=month_dt, day=1)
    pickup_tm = datetime(year=year_dt, month=month_dt, day=day_dt, hour=hour_tm, minute=minute_tm)
    pickup_seconds = (pickup_tm - month_start).total_seconds()/60
    return pickup_daytime, pickup_hour, pickup_weekday, pickup_quarter, pickup_seconds, pickup_holiday, pickup_weekend

def retrieve_model_artifacts()->str:
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config.read(os.path.join(parent_dir, "config", "config.ini"))

    application_path:str = config.get("settings", "application_path")
    model_artifacts_parent:str = config.get("ml-settings", "model_artifacts_path")
    model_artifacts_child:str = config.get("ml-settings", "duration_model_artifact")
    duration_model_path:str = os.path.join(
        application_path, model_artifacts_parent, model_artifacts_child
    )
    available_directories:list = [os.path.join(duration_model_path, file) for file in os.listdir(duration_model_path)]
    latest_modified_directory:str = max(available_directories, key=os.path.getmtime)
    return latest_modified_directory

def ai_advisor():
    # Set web-page and application title(s)
    st.set_page_config(page_title="AI advisor")
    st.title("Your AI search engine for taxi tips!")
    st.write("To trigger the converstation please type a greeding word to the input box on the bottom of the page (i.e. Hi)")
    # Establish connection to redis key-value store
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Set up triggering words
    triggering_appriciation_words = ["Thanks", "Thank you", "You are the best", "helpful"]
    triggering_advice_words = ["from", "to", "on", "at"]

    # Accept user input
    if prompt := st.chat_input("What's up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        if all(word.lower() in str.lower(st.session_state.messages[-1]["content"]) for word in triggering_advice_words):

            # Define regular expressions to capture the required information
            from_pattern = r'from\s+(.*?)\s+to'
            to_pattern = r'(?:.*\bto\s)(.*?)(?=\s*on\b)'
            day_pattern = r"on\s(\w+)\sat"
            time_pattern = r"at\s([\d:]+)"
            matching_time = r'\d{2}:\d{2}'

            # Extract information using the defined patterns
            pickup_location = re.search(from_pattern, prompt, re.IGNORECASE).group(1).strip()
            pickup_location = pickup_location.split(',')[0] if ',' in pickup_location else pickup_location
            dropoff_location = re.search(to_pattern, prompt, re.IGNORECASE).group(1).strip()
            dropoff_location = dropoff_location.split(',')[0] if ',' in dropoff_location else dropoff_location
            day = re.search(day_pattern, prompt, re.IGNORECASE).group(1).strip().lower()
            pickup_time = re.search(time_pattern, prompt, re.IGNORECASE).group(1).strip()

            print(f"From match: {pickup_location}")
            print(f"To match: {dropoff_location}")
            print(f"On match: {day}")
            print(f"At match: {pickup_time}")

            if (pickup_location and dropoff_location and day and pickup_time):
                if re.match(matching_time, pickup_time):
                    (go_from, go_to, neighborhood_from, neighborhood_to) = retrieve_trip_metadata(pickup_location, dropoff_location)
                    (trip_distance, centroid_distance) = compute_trip_and_centroid_distance((go_from, go_to, neighborhood_from, neighborhood_to))
                    (
                        pickup_daytime,
                        pickup_hour,
                        pickup_weekday,
                        pickup_quarter,
                        pickup_seconds,
                        pickup_holiday,
                        pickup_weekend
                    ) = retrieve_day_time_metadata(day, pickup_time)
                    features:np.ndarray = np.array(
                        [
                            trip_distance,
                            pickup_daytime,
                            pickup_hour,
                            pickup_weekday,
                            pickup_quarter,
                            pickup_seconds,
                            pickup_holiday,
                            pickup_weekend,
                            centroid_distance
                        ]
                    ).reshape(1,-1)
                    cache_key:str = str(features)
                    if r.exists(cache_key):
                        print("REDIS KEY EXISTS")
                        avg_predicted_trip_duration:float = float(r.get(cache_key))
                        print(avg_predicted_trip_duration)
                    else:
                        model_artifacts:str = retrieve_model_artifacts()
                        trip_type:str = "short_trip" if trip_distance < 30.0 else "long_trip"
                        models_path:str = os.path.join(model_artifacts, trip_type, "models")
                        predictions:list = []
                        for model in os.listdir(models_path):
                            regressor:Pipeline = joblib.load(os.path.join(models_path, model))
                            prediction = regressor.predict(features)[0]
                            predictions.append(prediction)
                        avg_predicted_trip_duration:float = np.round(np.mean(predictions), 2)
                        r.set(cache_key, str(avg_predicted_trip_duration))

                    delta = dt.timedelta(minutes=avg_predicted_trip_duration)
                    eta = eta_duration(pickup_time, avg_predicted_trip_duration)

                    # assistant_response = f"🚕 Your trip will have an average duration of {str(delta)}⏱ (eta: {eta}) and it will approximately cost ${np.round(predicted_cost, 2)}💸 (including taxes)."
                    assistant_response = f"🚕 Your trip will have an average duration of {str(delta)}⏱ (eta: {eta}) and it will approximately cost 12 dollaars."
                    with st.chat_message("assistant"):
                        mssg_placeholder, assistant_response = stream_simulation(assistant_response)
                    mssg_placeholder.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                else:
                    #st.error('Oops!', icon="🚨")
                    assistant_response = "🚨Oops! You didn't type your **pickup** time. Annotate *<TIME>* by using the keyword :blue[at] HH&#58;MM (24h scale)."
                    with st.chat_message("assistant"):
                        mssg_placeholder, assistant_response = stream_simulation(assistant_response)
                    mssg_placeholder.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            else:
                assistant_response = "🚨Oops! You didn't type your **pickup** and **dropoff** locations. Annotate *<LOCATION>* by using the keywords :blue[from] and :blue[to]."
                with st.chat_message("assistant"):
                    mssg_placeholder, assistant_response = stream_simulation(assistant_response)
                mssg_placeholder.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        elif len(st.session_state.messages) == 1:
            assistant_response = random.choice([
                "👋 Hello, I am your personal AI advisor for taxi trip's cost and duration. How can I help you?",
                "👋 Hi, there! Any taxi trip's cost and duration advice for today?",
                "👋 Greetings from your personal AI advisor. How can I help you?",
            ])
            with st.chat_message("assistant"):
                mssg_placeholder, assistant_response = stream_simulation(assistant_response)
            mssg_placeholder.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        elif any(word.lower() in str.lower(remove_punctuation(st.session_state.messages[-1]["content"])) for word in triggering_appriciation_words):
            assistant_response = "😎You're welcome. Have a great trip and keep safe!"
            with st.chat_message("assistant"):
                mssg_placeholder, assistant_response = stream_simulation(assistant_response)
            mssg_placeholder.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        else:
            assistant_response = "💔I am an ai taxi-trip advisor. Please write something like 'I would like to go :blue[from] <PLACE> :blue[to] <PLACE> :blue[on] <DAY>' :blue[at] <PICKUP TIME>' "
            with st.chat_message("assistant"):
                mssg_placeholder, assistant_response = stream_simulation(assistant_response)
            mssg_placeholder.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
if __name__ == '__main__':
    ai_advisor()
