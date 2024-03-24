import numpy as np
import pandas as pd
from keras.models import load_model
from joblib import load
from sklearn.preprocessing import LabelEncoder

def inverse_log_scale(data, epsilon=1e-7):
    return np.exp(data) - epsilon

model = load_model("./models/music_composition/composer.joblib")
instrument_encoder = load("./models/instrument_encoder.joblib")
genre_encoder = load("./models/genre_encoder.joblib")

def generate_seed_sequence():
    # This should return a numpy array matching the input shape expected by the model
    return np.random.rand(1, 5, 5) * 2 - 1 

def create_note_sequences(model, genre, num_sequences=10):
    genre_encoded = genre_encoder.transform([genre])
    genre_one_hot = np.zeros((1, genre_encoder.classes_.size))
    genre_one_hot[np.arange(1), genre_encoded] = 1
    
    seed_sequence = generate_seed_sequence()
    
    predicted_sequences = []
    
    for _ in range(num_sequences):
        predictions = model.predict([seed_sequence, np.array([np.zeros(genre_one_hot.shape)]), genre_one_hot])
        
        processed_predictions = predictions
        
        seed_sequence = processed_predictions
        
        predicted_sequences.append(processed_predictions)
    
    return predicted_sequences

user_genre = input("Please enter a genre (e.g., Blues, Jazz): ")

sequences = create_note_sequences(model, user_genre, num_sequences=10)

print("Generated sequences:", sequences)
