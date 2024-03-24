import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Masking, Dense, Embedding, concatenate, Reshape
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import tensorflow as tf
from joblib import dump


def log_scale(data, epsilon=1e-7):
    return np.log(float(data) + epsilon)  
  
def process_note_sequences(sequence_str):

    note_pattern = r"Note\(start=(.*?), end=(.*?), pitch=(.*?), velocity=(.*?)\),\s*(True|False)"
    matches = re.findall(note_pattern, sequence_str)

    processed_sequence = []    

    for note_tuple in matches:

        start_time, end_time, pitch, velocity, dummy_flag = note_tuple

        if dummy_flag == 'True':

            start_time = log_scale(start_time)          ##
            end_time = log_scale(end_time)              ##
            pitch = int(pitch) / 127.0                  ##  Normalization
            velocity = int(velocity) / 127.0            ##  
            dummy_flag = 1                              ##

        else:
            start_time = -1                             ##
            end_time = -1                               ##
            pitch = -1                                  ##  Marking dummy notes for masking
            velocity = -1                               ##
            dummy_flag = -1                             ##


        
        processed_sequence.append(list([start_time, end_time, pitch, velocity, dummy_flag]))

    return processed_sequence

def custom_loss(y_true, y_pred):

    mask = tf.cast(tf.not_equal(y_true[..., -1], -1), tf.float32)
    loss = tf.square(y_true - y_pred)
    mask_expanded = tf.expand_dims(mask, -1) 
    masked_loss = loss * mask_expanded
    return tf.reduce_mean(masked_loss)

df = pd.read_csv('test.csv')

instrument_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

df['Instrument'] = instrument_encoder.fit_transform(df['Instrument'])
df['Genre'] = genre_encoder.fit_transform(df['Genre'])

# One-hot encoding
instruments_encoded = to_categorical(df['Instrument'])
genres_encoded = to_categorical(df['Genre'])

X = np.array([process_note_sequences(seq) for seq in df['Sequence']])
y = np.array([process_note_sequences(seq) for seq in df['Labels']])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
instruments_train, instruments_val = train_test_split(instruments_encoded, test_size=0.2, random_state=42)
genres_train, genres_val = train_test_split(genres_encoded, test_size=0.2, random_state=42)

y_train_piano = y_train[:, :3, :]
y_train_guitar = y_train[:, 3:6, :]
y_train_bass = y_train[:, 6:9, :]
y_train_drum = y_train[:, 9:, :]

y_val_piano = y_val[:, :3, :]
y_val_guitar = y_val[:, 3:6, :]
y_val_bass = y_val[:, 6:9, :]
y_val_drum = y_val[:, 9:, :]


input_seq = Input(shape=(5, 5), name='input_seq')
input_inst = Input(shape=(instruments_encoded.shape[1],), name='input_inst')
input_genre = Input(shape=(genres_encoded.shape[1],), name='input_genre')

masking_layer = Masking(mask_value=-1)(input_seq)
lstm_layer = LSTM(64)(masking_layer)

concat_layer = concatenate([lstm_layer, input_inst, input_genre])

output_piano = Dense(units=3*5, activation='linear', name='output_piano')(lstm_layer)
output_guitar = Dense(units=3*5, activation='linear', name='output_guitar')(lstm_layer)
output_bass = Dense(units=3*5, activation='linear', name='output_bass')(lstm_layer)
output_drums = Dense(units=3*5, activation='linear', name='output_drums')(lstm_layer)

output_piano_reshaped = Reshape((3, 5), name='output_piano_reshaped')(output_piano)
output_guitar_reshaped = Reshape((3, 5), name='output_guitar_reshaped')(output_guitar)
output_bass_reshaped = Reshape((3, 5), name='output_bass_reshaped')(output_bass)
output_drums_reshaped = Reshape((3, 5), name='output_drums_reshaped')(output_drums)



model = Model(inputs=[input_seq, input_inst, input_genre], outputs=[output_piano_reshaped, output_guitar_reshaped, output_bass_reshaped, output_drums_reshaped])

model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

trained_model = model.fit([X_train, instruments_train, genres_train], [y_train_piano, y_train_guitar, y_train_bass, y_train_drum], 
                            validation_data=([X_val, instruments_val, genres_val], [y_val_piano, y_val_guitar, y_val_bass, y_val_drum]),
                            epochs=10, batch_size=64)


dump(trained_model, "./models/music_composition/composer.joblib")



