import os
import numpy as np
from music21 import converter, note, stream, chord
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Embedding, Flatten, Concatenate
from keras.optimizers import Adam

def getNotesAndAttributes(dataset_folder):
    notes_and_attributes = []  # List to hold notes along with instrument and genre

    for genre_folder in os.listdir(dataset_folder):
        genre_path = os.path.join(dataset_folder, genre_folder)

        for instrument_folder in os.listdir(genre_path):
            instrument_path = os.path.join(genre_path, instrument_folder)

            for file in os.listdir(instrument_path):
                if file.endswith('.mid') or file.endswith('.midi'):

                    midi = converter.parse(os.path.join(instrument_path, file))
                    print(f'Parsing {file} in {instrument_folder} for {genre_folder}')

                    notes = []                    
                    first_note_or_chord_found = False

                    for element in midi.flat.notesAndRests:
                        
                        if not first_note_or_chord_found:

                            if isinstance(element, (note.Note, chord.Chord)):
                                first_note_or_chord_found = True

                            else:
                                continue

                        if isinstance(element, note.Rest):
                            notes.append("rest")
                            
                        if isinstance(element, note.Note):
                            notes.append(str(element.pitch))

                        elif isinstance(element, chord.Chord):
                            notes.append('.'.join(str(n) for n in element.pitches))

                    #!!!!!!!!!!!!!!! Add 1 more input feature that checks if instrument exists in the midi file
                    # Change data processing. It should 

                    notes_and_attributes.append((notes, instrument_folder, genre_folder))

    return notes_and_attributes


seq_length = 15  # Length of input sequences    
note_vocab_size = 500  # Number of unique notes/chords
embedding_dim = 64  # Dimensionality of embedding vectors
instrument_vocab_size = 10  # Number of unique instruments
genre_vocab_size = 5  # Number of unique genres

# Input layers
note_input = Input(shape=(seq_length, 1), name='note_input')
instrument_input = Input(shape=(1,), name='instrument_input')
genre_input = Input(shape=(1,), name='genre_input')


#!!!!!!!!!!!!!!!!Include that new feature in the layers

# Embeddings
instrument_embedding = Embedding(input_dim=instrument_vocab_size, output_dim=embedding_dim)(instrument_input)
genre_embedding = Embedding(input_dim=genre_vocab_size, output_dim=embedding_dim)(genre_input)
instrument_flat = Flatten()(instrument_embedding)
genre_flat = Flatten()(genre_embedding)

# Concatenate note input with flattened instrument and genre embeddings
concat = Concatenate()([note_input, instrument_flat, genre_flat])

# Shared LSTM layers
lstm_shared = LSTM(64, return_sequences=True)(concat)
lstm_shared = Dropout(0.3)(lstm_shared)
lstm_shared = LSTM(64)(lstm_shared)
lstm_shared = Dropout(0.3)(lstm_shared)

# Instrument-specific branches
guitar_branch = Dense(128, activation='relu')(lstm_shared)
guitar_output = Dense(note_vocab_size, activation='softmax', name='guitar_output')(guitar_branch)

bass_branch = Dense(128, activation='relu')(lstm_shared)
bass_output = Dense(note_vocab_size, activation='softmax', name='bass_output')(bass_branch)

piano_branch = Dense(128, activation='relu')(lstm_shared)
piano_output = Dense(note_vocab_size, activation='softmax', name='piano_output')(piano_branch)

# Construct the model with multiple outputs
model = Model(inputs=[note_input, instrument_input, genre_input], outputs=[guitar_output, bass_output, piano_output])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary to verify architecture
model.summary()

model.fit(x=[input_notes, input_instruments, input_genres], y=[target_guitar, target_bass, target_piano], epochs=50, batch_size=64)