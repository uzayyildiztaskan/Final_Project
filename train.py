import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Masking, Dense, Bidirectional, concatenate, Reshape, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import tensorflow as tf
from sources.names import essential_instrument_names, genres
from keras.callbacks import ReduceLROnPlateau
import pickle
import ast

def log_scale(data, epsilon=1e-7):
    return np.log(float(data) + epsilon)  
  
def process_note_sequences(sequence_str):

    tuples = ast.literal_eval(sequence_str)

    processed_sequence = []    

    for note_tuple in tuples:

        if len(note_tuple) == 2:        # Label sequence

            instrument, note = note_tuple
            duration, pitch, velocity = note

        else :
            duration, pitch, velocity = note_tuple

        if pitch  != -2:

            duration = duration_enumarator[duration]                ##
            pitch = pitch_enumarator[pitch]                         ##  
                                                                    ##
            duration = duration / len(duration_enumarator)          ##  Normalization
            pitch = pitch / len(pitch_enumarator)                   ##
                                                                    ##
            velocity = int(velocity) / 127.0                        ##

        else:
            duration = -2                              ##
            pitch = -2                                 ##  Marking dummy notes for masking
            velocity = -2                              ##


        
        processed_sequence.append(list([duration, pitch, velocity]))

    return processed_sequence

def custom_loss(y_true, y_pred):

    mask = tf.cast(tf.not_equal(y_true[..., -1], -2), tf.float32)
    loss = tf.square(y_true - y_pred)
    mask_expanded = tf.expand_dims(mask, -1) 
    masked_loss = loss * mask_expanded
    return tf.reduce_mean(masked_loss)

with open('./enumarators/duration_enumarator.pickle', 'rb') as handle:
    duration_enumarator = pickle.load(handle)

with open('./enumarators/pitch_enumarator.pickle', 'rb') as handle:
    pitch_enumarator = pickle.load(handle)

with open('./enumarators/tempo_enumarator.pickle', 'rb') as handle:
    tempo_enumarator = pickle.load(handle)

df = pd.read_csv('data_test.csv')

sequence_length = 10

instrument_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

instrument_encoder.fit(essential_instrument_names)
genre_encoder.fit(genres)

df['Instrument'] = instrument_encoder.transform(df['Instrument'])
df['Genre'] = genre_encoder.transform(df['Genre'])

# One-hot encoding
instruments_encoded = to_categorical(df['Instrument'], num_classes = len(essential_instrument_names))
genres_encoded = to_categorical(df['Genre'], num_classes = len(genres))

X = np.array([process_note_sequences(seq) for seq in df['Sequence']])
y = np.array([process_note_sequences(seq) for seq in df['Labels']])
tempos = np.array([tempo_enumarator[tempo] / len(tempo_enumarator) for tempo in df['Tempo']])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
instruments_train, instruments_val = train_test_split(instruments_encoded, test_size=0.2, random_state=42)
genres_train, genres_val = train_test_split(genres_encoded, test_size=0.2, random_state=42)
tempos_train, tempos_val = train_test_split(tempos, test_size=0.2, random_state=42)


y_train_bass = y_train[:, 0, :]
y_train_drum = y_train[:, 1, :]
y_train_guitar = y_train[:, 2, :]
y_train_piano = y_train[:, 3:, :]


y_val_bass = y_val[:, 0, :]
y_val_drum = y_val[:, 1, :]
y_val_guitar = y_val[:, 2, :]
y_val_piano = y_val[:, 3:, :]


input_seq = Input(shape=(sequence_length, 3), name='input_seq')
input_inst = Input(shape=(instruments_encoded.shape[1],), name='input_inst')
input_genre = Input(shape=(genres_encoded.shape[1],), name='input_genre')
input_tempo = Input(shape=(1,), name="input_tempo")

masking_layer = Masking(mask_value=-2)(input_seq)

lstm_layer_1 = LSTM(64, return_sequences=True)(masking_layer)
dropout_1 = Dropout(0.2)(lstm_layer_1)

lstm_layer_2 = LSTM(128, return_sequences=True)(dropout_1)
dropout_2 = Dropout(0.2)(lstm_layer_2)

lstm_layer_3 = LSTM(64, return_sequences=True)(dropout_2)
dropout_3 = Dropout(0.2)(lstm_layer_3)

lstm_layer_4 = LSTM(64, return_sequences=False)(dropout_3)
dropout_4 = Dropout(0.2)(lstm_layer_4)

concat_layer = concatenate([dropout_4, input_inst, input_genre, input_tempo])

output_bass = Dense(units=1*3, activation='softmax', name='output_bass')(concat_layer)
output_drum = Dense(units=1*3, activation='softmax', name='output_drum')(concat_layer)
output_guitar = Dense(units=1*3, activation='softmax', name='output_guitar')(concat_layer)
output_piano = Dense(units=1*3, activation='softmax', name='output_piano')(concat_layer)

output_bass_reshaped = Reshape((1, 3), name='output_bass_reshaped')(output_bass)
output_drum_reshaped = Reshape((1, 3), name='output_drum_reshaped')(output_drum)
output_guitar_reshaped = Reshape((1, 3), name='output_guitar_reshaped')(output_guitar)
output_piano_reshaped = Reshape((1, 3), name='output_piano_reshaped')(output_piano)


model = Model(inputs=[input_seq, input_inst, input_genre, input_tempo], outputs=[output_bass_reshaped, output_drum_reshaped, output_guitar_reshaped, output_piano_reshaped])

model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

model.fit([X_train, instruments_train, genres_train, tempos_train], [y_train_bass, y_train_drum, y_train_guitar, y_train_piano], 
                            validation_data=([X_val, instruments_val, genres_val, tempos_val], [y_val_bass, y_val_drum, y_val_guitar, y_val_piano]),
                            epochs=10, batch_size=8, callbacks=[reduce_lr])


model.save("./models/music_composition/composer.h5")



