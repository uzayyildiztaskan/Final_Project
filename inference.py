import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from pretty_midi import Note
from sources.names import essential_instrument_names, genres
from keras.utils import to_categorical
import tensorflow as tf
import pretty_midi
import random
import ast
import pickle
from sources.random_sequences import all_seeds

def custom_loss(y_true, y_pred):

    mask = tf.cast(tf.not_equal(y_true[..., -1], -2), tf.float32)
    loss = tf.square(y_true - y_pred)
    mask_expanded = tf.expand_dims(mask, -1) 
    masked_loss = loss * mask_expanded
    return tf.reduce_mean(masked_loss)

def inverse_log_scale(data, epsilon=1e-7):
    return np.exp(data) - epsilon

def log_scale(data, epsilon=1e-7):
    return np.log(float(data) + epsilon)  

def process_random_seed(instrument_seed, instrument_sequence, instrument_time, tempo):

    instrument_seed = ast.literal_eval(instrument_seed)[-5:]

    for note_obj in instrument_seed:

        duration = note_obj[0].split("/")

        numerator, denominator = duration

        duration = (float(numerator) / float(denominator)) * float(120.0 / tempo)

        if "," in note_obj[1]:

            chord = note_obj[1].split(",")

            for note in chord:

                pitch_value = ast.literal_eval(note)

                instrument_sequence.append(Note(start = instrument_time, end = instrument_time + duration, pitch = pitch_value, velocity = note_obj[2]))

        else:

            pitch_value = ast.literal_eval(note_obj[1])

            instrument_sequence.append(Note(start = instrument_time, end = instrument_time + duration, pitch = pitch_value, velocity = note_obj[2]))

        instrument_time += duration

    return instrument_sequence, instrument_time


def create_note_sequences(model, genre, num_sequences=100):

    genre_index = genre_encoder.transform([genre])[0]
    genre = to_categorical(genre_encoder.transform([genre]), num_classes = len(genres))

    bass_tempo = 30
    drum_tempo = 15
    guitar_tempo = 60
    piano_tempo = 30

    bass_sequences = []
    drum_sequences = []
    guitar_sequences = []
    piano_sequences = []

    bass_time = 0
    drum_time = 0
    guitar_time = 0
    piano_time = 0

    bass_seed = []
    drum_seed = []
    guitar_seed = []
    piano_seed = []
    
    for time_step in range(num_sequences):  

        if time_step % 3 == 0:

            random_index_bass = random.randint(0, len(all_seeds[genre_index][0]) - 1)
            random_index_drum = random.randint(0, len(all_seeds[genre_index][1]) - 1)
            random_index_guitar = random.randint(0, len(all_seeds[genre_index][2]) - 1)
            random_index_piano = random.randint(0, len(all_seeds[genre_index][3]) - 1)

            bass_seed = process_note_sequences(all_seeds[genre_index][0][random_index_bass])
            drum_seed = process_note_sequences(all_seeds[genre_index][1][random_index_drum])
            guitar_seed = process_note_sequences(all_seeds[genre_index][2][random_index_guitar])
            piano_seed = process_note_sequences(all_seeds[genre_index][3][random_index_piano])

            bass_sequences, bass_time = process_random_seed(all_seeds[genre_index][0][random_index_bass], bass_sequences, bass_time, bass_tempo)

            drum_sequences, drum_time = process_random_seed(all_seeds[genre_index][1][random_index_drum], drum_sequences, drum_time, drum_tempo)

            guitar_sequences, guitar_time = process_random_seed(all_seeds[genre_index][2][random_index_guitar], guitar_sequences, guitar_time, guitar_tempo)

            piano_sequences, piano_time = process_random_seed(all_seeds[genre_index][3][random_index_piano], piano_sequences, piano_time, piano_tempo)


        predictions = model.predict([bass_seed, genre, drum_seed, genre, guitar_seed, genre, piano_seed, genre])

        bass_prediction = np.array(predictions)[0]
        drum_prediction = np.array(predictions)[1]
        guitar_prediction = np.array(predictions)[2]
        piano_prediction = np.array(predictions)[3]

        bass_seed = np.concatenate([bass_seed, bass_prediction], axis=1)[:, 1:, :]
        drum_seed = np.concatenate([drum_seed, drum_prediction], axis=1)[:, 1:, :]
        guitar_seed = np.concatenate([guitar_seed, guitar_prediction], axis=1)[:, 1:, :]
        piano_seed = np.concatenate([piano_seed, piano_prediction], axis=1)[:, 1:, :]

        bass_sequences, bass_time = process_prediction(predictions[0][0][0], bass_sequences, bass_time, bass_tempo)

        drum_sequences, drum_time = process_prediction(predictions[1][0][0], drum_sequences, drum_time, drum_tempo)

        guitar_sequences, guitar_time = process_prediction(predictions[2][0][0], guitar_sequences, guitar_time, guitar_tempo)

        piano_sequences, piano_time = process_prediction(predictions[3][0][0], piano_sequences, piano_time, piano_tempo)


    return bass_sequences, drum_sequences, guitar_sequences, piano_sequences

def create_midi_file(sequences, filename="./outputs/composed_music.mid", bpm=120, time_signature=(4, 4), key_signature=0):
    
    midi_file = pretty_midi.PrettyMIDI()    

    instrument_bass = pretty_midi.Instrument(program=35)
    
    instrument_bass.notes.extend(sequences[0])

    instrument_drum = pretty_midi.Instrument(program=0, is_drum=True)

    instrument_drum.notes.extend(sequences[1])

    instrument_guitar = pretty_midi.Instrument(program=29)

    instrument_guitar.notes.extend(sequences[2])

    instrument_piano = pretty_midi.Instrument(program=0)

    instrument_piano.notes.extend(sequences[3])    

    midi_file.instruments.append(instrument_guitar)
    midi_file.instruments.append(instrument_piano)
    midi_file.instruments.append(instrument_bass)
    midi_file.instruments.append(instrument_drum)
    

    midi_file.instruments[0].control_changes = [pretty_midi.ControlChange(number=0, value=0, time=0)]
    
    ts_change = pretty_midi.TimeSignature(time_signature[0], time_signature[1], 0)
    midi_file.time_signature_changes.append(ts_change)

    ks_change = pretty_midi.KeySignature(key_signature, 0)
    midi_file.key_signature_changes.append(ks_change)

    midi_file.write(filename)

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

    return np.array([processed_sequence])


def process_prediction(prediction, instrument_sequence, instrument_time, tempo):

    start_time = instrument_time

    pitch = round(prediction[1] * len(pitch_enumarator))

    pitch = pitch_denumarator[pitch]

    duration = "1/4"

    duration = duration.split("/")

    numerator, denominator = duration

    duration = (float(numerator) / float(denominator)) * float(120.0 / tempo)

    if random.random() < 0.3:

        return instrument_sequence, instrument_time + duration

    velocity = round(prediction[2] * 127.0)

    if "," in pitch:

        chord = pitch.split(",")

        for note in chord:

            pitch_value = ast.literal_eval(note)

            instrument_sequence.append(Note(start = start_time, end = start_time + duration, pitch = pitch_value, velocity = velocity))

    else:

        pitch_value = ast.literal_eval(pitch)

        instrument_sequence.append(Note(start = start_time, end = start_time + duration, pitch = pitch_value, velocity = velocity))

    instrument_time += duration

    return instrument_sequence, instrument_time

def process_final_sequences(sequences):

    for instrument in sequences:

        excess_time = instrument[0].start

        for note in instrument:

            note.start = note.start - excess_time
            note.end = note.end - excess_time

    return sequences


with open('./enumarators/duration_enumarator.pickle', 'rb') as handle:
    duration_enumarator = pickle.load(handle)

with open('./enumarators/pitch_enumarator.pickle', 'rb') as handle:
    pitch_enumarator = pickle.load(handle)

with open('./enumarators/tempo_enumarator.pickle', 'rb') as handle:
    tempo_enumarator = pickle.load(handle)

duration_denumarator = {i: note for note, i in duration_enumarator.items()}
pitch_denumarator = {i: note for note, i in pitch_enumarator.items()}


model = load_model("./models/music_composition/composer.h5", custom_objects={"custom_loss": custom_loss})
instrument_encoder = LabelEncoder().fit(essential_instrument_names)
genre_encoder = LabelEncoder().fit(genres)

user_genre = "Blues"

sequences = create_note_sequences(model, user_genre, num_sequences=10)

sequences = process_final_sequences(sequences)

create_midi_file(sequences)
