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

def pad_values(X_values, y_values, largest_dimension):

    counter = 0

    padded_X_values = []
    padded_y_values = []

    for x in X_values:

        if x.shape[0] < largest_dimension:

            if counter % 2 == 0:

                padding_values = np.full((largest_dimension - x.shape[0], 10, 3), -2)                

            else:

                padding_values = np.full((largest_dimension - x.shape[0], 14), -2)                

            x = np.append(x, padding_values, axis=0)

        padded_X_values.append(x)
        
        counter += 1

    for y in y_values:

        if y.shape[0] < largest_dimension:

            padding_values = np.full((largest_dimension - y.shape[0], 1, 3), -2)

            y = np.append(y, padding_values, axis=0)
        
        padded_y_values.append(y)

    return padded_X_values, padded_y_values


with open('./enumarators/duration_enumarator.pickle', 'rb') as handle:
    duration_enumarator = pickle.load(handle)

with open('./enumarators/pitch_enumarator.pickle', 'rb') as handle:
    pitch_enumarator = pickle.load(handle)


sequence_length = 10

instrument_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

instrument_encoder.fit(essential_instrument_names)
genre_encoder.fit(genres)

########### Bass ###########

df_bass = pd.read_csv('./processed_excel_data/Bass.csv')
df_bass['Genre'] = genre_encoder.transform(df_bass['Genre'])

X_bass = np.array([process_note_sequences(seq) for seq in df_bass['Sequence']])
y_bass = np.array([process_note_sequences(seq) for seq in df_bass['Label']])

genres_encoded_bass = to_categorical(df_bass['Genre'], num_classes = len(genres))

X_train_bass, X_val_bass, y_train_bass, y_val_bass = train_test_split(X_bass, y_bass, test_size=0.2, random_state=42)
genres_train_bass, genres_val_bass = train_test_split(genres_encoded_bass, test_size=0.2, random_state=42)

########### Drum ###########

df_drum = pd.read_csv('./processed_excel_data/Drum.csv')
df_drum['Genre'] = genre_encoder.transform(df_drum['Genre'])

X_drum = np.array([process_note_sequences(seq) for seq in df_drum['Sequence']])
y_drum = np.array([process_note_sequences(seq) for seq in df_drum['Label']])

genres_encoded_drum = to_categorical(df_drum['Genre'], num_classes = len(genres))

X_train_drum, X_val_drum, y_train_drum, y_val_drum = train_test_split(X_drum, y_drum, test_size=0.2, random_state=42)
genres_train_drum, genres_val_drum = train_test_split(genres_encoded_drum, test_size=0.2, random_state=42)

########### Guitar ###########

df_guitar = pd.read_csv('./processed_excel_data/Guitar.csv')
df_guitar['Genre'] = genre_encoder.transform(df_guitar['Genre'])

X_guitar = np.array([process_note_sequences(seq) for seq in df_guitar['Sequence']])
y_guitar = np.array([process_note_sequences(seq) for seq in df_guitar['Label']])

genres_encoded_guitar = to_categorical(df_guitar['Genre'], num_classes = len(genres))

X_train_guitar, X_val_guitar, y_train_guitar, y_val_guitar = train_test_split(X_guitar, y_guitar, test_size=0.2, random_state=42)
genres_train_guitar, genres_val_guitar = train_test_split(genres_encoded_guitar, test_size=0.2, random_state=42)

########### Piano ###########

df_piano = pd.read_csv('./processed_excel_data/piano.csv')
df_piano['Genre'] = genre_encoder.transform(df_piano['Genre'])

X_piano = np.array([process_note_sequences(seq) for seq in df_piano['Sequence']])
y_piano = np.array([process_note_sequences(seq) for seq in df_piano['Label']])

genres_encoded_piano = to_categorical(df_piano['Genre'], num_classes = len(genres))

X_train_piano, X_val_piano, y_train_piano, y_val_piano = train_test_split(X_piano, y_piano, test_size=0.2, random_state=42)
genres_train_piano, genres_val_piano = train_test_split(genres_encoded_piano, test_size=0.2, random_state=42)


input_seq_bass = Input(shape=(sequence_length, 3), name='input_seq_bass')
input_genre_bass = Input(shape=(genres_encoded_bass.shape[1],), name='input_genre_bass')

input_seq_drum = Input(shape=(sequence_length, 3), name='input_seq_drum')
input_genre_drum = Input(shape=(genres_encoded_drum.shape[1],), name='input_genre_drum')

input_seq_guitar = Input(shape=(sequence_length, 3), name='input_seq_guitar')
input_genre_guitar = Input(shape=(genres_encoded_guitar.shape[1],), name='input_genre_guitar')

input_seq_piano = Input(shape=(sequence_length, 3), name='input_seq_piano')
input_genre_piano = Input(shape=(genres_encoded_piano.shape[1],), name='input_genre_piano')

masking_layer_bass = Masking(mask_value=-2)(input_seq_bass)
masking_layer_drum = Masking(mask_value=-2)(input_seq_drum)
masking_layer_guitar = Masking(mask_value=-2)(input_seq_guitar)
masking_layer_piano = Masking(mask_value=-2)(input_seq_piano)

masking_layer_genre_bass = Masking(mask_value=-2)(input_genre_bass)
masking_layer_genre_drum = Masking(mask_value=-2)(input_genre_drum)
masking_layer_genre_guitar = Masking(mask_value=-2)(input_genre_guitar)
masking_layer_genre_piano = Masking(mask_value=-2)(input_genre_piano)

lstm_layer_1_bass = LSTM(128, return_sequences=True)(masking_layer_bass)
dropout_1_bass = Dropout(0.2)(lstm_layer_1_bass)
lstm_layer_2_bass = LSTM(64, return_sequences=True)(dropout_1_bass)
dropout_2_bass = Dropout(0.4)(lstm_layer_2_bass)
lstm_layer_3_bass = LSTM(32, return_sequences=False)(dropout_2_bass)

lstm_layer_1_drum = LSTM(128, return_sequences=True)(masking_layer_drum)
dropout_1_drum = Dropout(0.2)(lstm_layer_1_drum)
lstm_layer_2_drum = LSTM(64, return_sequences=True)(dropout_1_drum)
dropout_2_drum = Dropout(0.4)(lstm_layer_2_drum)
lstm_layer_3_drum = LSTM(32, return_sequences=False)(dropout_2_drum)

lstm_layer_1_guitar = LSTM(128, return_sequences=True)(masking_layer_guitar)
dropout_1_guitar = Dropout(0.2)(lstm_layer_1_guitar)
lstm_layer_2_guitar = LSTM(64, return_sequences=True)(dropout_1_guitar)
dropout_2_guitar = Dropout(0.4)(lstm_layer_2_guitar)
lstm_layer_3_guitar = LSTM(32, return_sequences=False)(dropout_2_guitar)

lstm_layer_1_piano = LSTM(128, return_sequences=True)(masking_layer_piano)
dropout_1_piano = Dropout(0.2)(lstm_layer_1_piano)
lstm_layer_2_piano = LSTM(64, return_sequences=True)(dropout_1_piano)
dropout_2_piano = Dropout(0.4)(lstm_layer_2_piano)
lstm_layer_3_piano = LSTM(32, return_sequences=False)(dropout_2_piano)


concat_layer_bass = concatenate([lstm_layer_3_bass, masking_layer_genre_bass])
concat_layer_drum = concatenate([lstm_layer_3_drum, masking_layer_genre_drum])
concat_layer_guitar = concatenate([lstm_layer_3_guitar, masking_layer_genre_guitar])
concat_layer_piano = concatenate([lstm_layer_3_piano, masking_layer_genre_piano])

output_bass = Dense(units=1*3, name='output_bass', activation='linear')(concat_layer_bass)
output_drum = Dense(units=1*3, name='output_drum', activation='linear')(concat_layer_drum)
output_guitar = Dense(units=1*3, name='output_guitar', activation='linear')(concat_layer_guitar)
output_piano = Dense(units=1*3, name='output_piano', activation='linear')(concat_layer_piano)

output_bass_reshaped = Reshape((1, 3), name='output_bass_reshaped')(output_bass)
output_drum_reshaped = Reshape((1, 3), name='output_drum_reshaped')(output_drum)
output_guitar_reshaped = Reshape((1, 3), name='output_guitar_reshaped')(output_guitar)
output_piano_reshaped = Reshape((1, 3), name='output_piano_reshaped')(output_piano)


model = Model(inputs=[input_seq_bass, input_genre_bass, input_seq_drum, input_genre_drum, input_seq_guitar, input_genre_guitar, input_seq_piano, input_genre_piano], 
              outputs=[output_bass_reshaped, output_drum_reshaped, output_guitar_reshaped, output_piano_reshaped])

model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

X_train = [X_train_bass, genres_train_bass, X_train_drum, genres_train_drum, X_train_guitar, genres_train_guitar, X_train_piano, genres_train_piano]
y_train = [y_train_bass, y_train_drum, y_train_guitar, y_train_piano]

X_val = [X_val_bass, genres_val_bass, X_val_drum, genres_val_drum, X_val_guitar, genres_val_guitar, X_val_piano, genres_val_piano]
y_val = [y_val_bass, y_val_drum, y_val_guitar, y_val_piano]

largest_dimension_train = max(array.shape[0] for array in y_train)
largest_dimension_val = max(array.shape[0] for array in y_val)

X_train, y_train = pad_values(X_train, y_train, largest_dimension_train)
X_val, y_val = pad_values(X_val, y_val, largest_dimension_val)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8, callbacks=[reduce_lr])


model.save("./models/music_composition/composer.h5")



