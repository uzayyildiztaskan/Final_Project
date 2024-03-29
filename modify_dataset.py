import pretty_midi
import pandas as pd
import os
from sources.names import instrument_names, essential_instrument_names
from tqdm import tqdm
import logging
import csv
import pickle
import numpy as np
from fractions import Fraction


def get_essential_instrument_name(instrument_name):

    if "Piano" in instrument_name:
        return "Piano"
    
    if "Guitar" in instrument_name:
        return "Guitar"
    
    if "Bass" in instrument_name:
        return "Bass"
    
    if "Drum" in instrument_name:
        return "Drum"
    
def find_closest_fraction(target_fraction, max_digits=1):

    closest_fraction = None
    smallest_difference = float('inf')

    for numerator in range(1, 10**max_digits):
        for denominator in range(1, 10**max_digits):
            current_fraction = Fraction(numerator, denominator)
            current_difference = abs(current_fraction - target_fraction)

            if current_difference < smallest_difference:
                smallest_difference = current_difference
                closest_fraction = current_fraction

    return closest_fraction

def process_midi_file(csvwriter, midi_data, genre, threshold, sequence_length, max_chord_notes = 4):

    instrument_sequence_list = []

    tempo = round(midi_data.get_tempo_changes()[1][0])

    if tempo not in tempos:
        tempos.append(tempo)

    for instrument in midi_data.instruments:

        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

        if instrument_name in instrument_names.values():

            essential_instrument_name = get_essential_instrument_name(instrument_name)

            if len(instrument_sequence_list) == 0 or essential_instrument_name not in instrument_sequence_list:

                instrument_sequence_list.append(essential_instrument_name)
                
                current_sequence = []

                chord = []

                chord_end_times = []
                chord_start_times = []

                for note in instrument.notes:

                    quarter_duration = (tempo / 120) * (note.end - note.start)

                    fraction = Fraction(quarter_duration).limit_denominator()

                    fraction = find_closest_fraction(fraction)

                    duration = f"{fraction.numerator}/{fraction.denominator}"

                    if len(chord) == 0 or (note.start - chord_end_times[-1] <= threshold and len(chord) < max_chord_notes):

                        chord.append((duration, note.pitch, note.velocity))
                        chord_start_times.append(note.start)
                        chord_end_times.append(note.end)

                    else:

                        chord_pitch = ','.join(str(chord_note[1]) for chord_note in chord)

                        if chord_pitch not in pitches:
                            pitches.append(chord_pitch)

                        if duration not in durations:
                            durations.append(duration)
                        
                        current_sequence.append((chord[0][0], chord_pitch, chord[0][2]))

                        chord = [(duration, note.pitch, note.velocity)]
                        chord_start_times = [note.start]
                        chord_end_times = [note.end]

                    if len(current_sequence) == sequence_length:

                        labels = find_labels(midi_data, chord_start_times[-1], chord_end_times[-1], tempo, threshold, max_chord_notes)

                        if len(labels) != 0:

                            csvwriter.writerow([essential_instrument_name, genre, tempo, current_sequence, labels])

                            current_sequence.pop(0)                            
            
def find_labels(midi_data, start_time, end_time, tempo, threshold, max_chord_notes):

    labels = []
    found_labels = []

    for instrument in midi_data.instruments:

        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

        if instrument_name in instrument_names.values():

            essential_instrument_name = get_essential_instrument_name(instrument_name)

            chord = []

            chord_end_times = []

            for note in instrument.notes:
                    
                    if note.start >= start_time:

                        if note.start - end_time > 1:

                            quarter_duration = (tempo / 120) * (note.start - end_time)

                            fraction = Fraction(quarter_duration).limit_denominator()

                            fraction = find_closest_fraction(fraction)

                            duration = f"{fraction.numerator}/{fraction.denominator}"

                            labels.append([essential_instrument_name, (duration, "-1", 60)])

                            found_labels.append(essential_instrument_name)

                            if duration not in durations:
                                durations.append(duration)

                            break

                        quarter_duration = (tempo / 120) * (note.end - note.start)

                        fraction = Fraction(quarter_duration).limit_denominator()

                        fraction = find_closest_fraction(fraction)

                        duration = f"{fraction.numerator}/{fraction.denominator}"

                        if len(chord) == 0 or (note.start - chord_end_times[-1] <= threshold and len(chord) < max_chord_notes):

                            chord.append((duration, note.pitch, note.velocity))
                            chord_end_times.append(note.end)

                        else:

                            chord_pitch = ','.join(str(chord_note[1]) for chord_note in chord)

                            if chord_pitch not in pitches:
                                pitches.append(chord_pitch)
                            
                            if duration not in durations:
                                durations.append(duration)
                            
                            labels.append([essential_instrument_name, (chord[0][0], chord_pitch, chord[0][2])])

                            found_labels.append(essential_instrument_name)

                            break

    if len(found_labels) == 0:
        return []

    labels_in_order = []

    for ein in essential_instrument_names:
        
        if ein in found_labels:
            
            labels_in_order.append(labels[found_labels.index(ein)])

        else:

            labels_in_order.append([ein, (-2, -2, -2)])

    return labels_in_order

def create_csv_from_dataset(dataset_path, output_csv_path, sequence_length, max_track_amount_per_genre, threshold):

    with open(output_csv_path, 'w', newline='') as csvfile:
        # Create a writer object
        csvwriter = csv.writer(csvfile)
        
        csvwriter.writerow(['Instrument', 'Genre', 'Tempo', 'Sequence', 'Labels'])
        
        total_track_amount = 1

        for genre_folder in tqdm(os.listdir(dataset_path)):
            genre_folder_path = os.path.join(dataset_path, genre_folder)

            genre_track_amount = 1
            
            for midi_file_name in tqdm(os.listdir(genre_folder_path)):
                midi_file_path = os.path.join(genre_folder_path, midi_file_name)

                if(genre_track_amount > max_track_amount_per_genre):
                    break

                try:
                    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

                except: 
                    print(f"Error processing {midi_file_path}")
                    continue

                process_midi_file(csvwriter, midi_data, genre_folder, threshold, sequence_length)

                genre_track_amount += 1
            
            total_track_amount += genre_track_amount
            logging.info(f"Created {genre_folder} genre folder with {genre_track_amount} tracks in it.")
        
        logging.info(f"Dataset converted with a total number of {total_track_amount} tracks.")
        

logging.basicConfig(filename='logs/modify_dataset.log', level=logging.INFO)



pitches = []
durations = []
tempos = []

pitches.append("-1")

create_csv_from_dataset(dataset_path = "./genre_classified_dataset", output_csv_path = "data_test.csv", sequence_length = 10, max_track_amount_per_genre = 10, threshold = 0)

pitches_dict = {pitch: i for i, pitch in enumerate(sorted(pitches))}

durations_dict = {duration: i for i, duration in enumerate(sorted(durations))}

tempos_dict = {tempo: i for i, tempo in enumerate(sorted(tempos))}

with open('./enumarators/pitch_enumarator.pickle', 'wb') as handle:
    pickle.dump(pitches_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./enumarators/duration_enumarator.pickle', 'wb') as handle:
    pickle.dump(durations_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./enumarators/tempo_enumarator.pickle', 'wb') as handle:
    pickle.dump(tempos_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)