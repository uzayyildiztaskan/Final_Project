import pretty_midi
import pandas as pd
import os
from sources.names import instrument_names
from tqdm import tqdm
import logging

def get_bar_as_threshold(midi_data):

    try:

        time_signature = midi_data.time_signature_changes[0]

    except:

        time_signature = pretty_midi.TimeSignature(numerator = 4, denominator = 4, time = 0)

    try:

        tempo = round(midi_data.get_tempo_changes()[1][0])
    
    except: 

        tempo = 120

    bar_duration = (60 / tempo * (time_signature.denominator / 4) * time_signature.numerator)

    return bar_duration

def process_midi_file(midi_data, genre, sequence_length, threshold, minimum_related_note_amount):
    """
    Processes a single MIDI file to extract note sequences and labels for each instrument,
    considering the threshold for temporal gaps between notes.
    """

    all_sequences = []

    for instrument in midi_data.instruments:

        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        
        if(instrument_name in instrument_list):
            notes = sorted(instrument.notes, key=lambda note: note.start)
            
            current_sequence = [(notes[0], True)]

            note_index = 1
            
            while note_index < len(notes):

                if (notes[note_index].start - notes[note_index - 1].end) <= threshold:
                    
                    is_valid = True     # True if note is valid, false if note is a dummy note
                    current_sequence.append((notes[note_index], is_valid))
                    
                    if (len(current_sequence) == sequence_length):

                        labels = find_labels_within_threshold(midi_data, instrument_name, notes, current_sequence[-1][0].start, threshold)
                        all_sequences.append([instrument_name, genre, current_sequence, labels])
                        current_sequence.pop(0)

                elif (notes[note_index].start - notes[note_index - 1].end) > threshold:
                    
                    if (find_related_note_amount(current_sequence) < minimum_related_note_amount): # Reset sequence

                        is_valid = True
                        current_sequence = [(notes[note_index], is_valid)]
                    
                    else:
                        
                        current_sequence = apply_padding(current_sequence, sequence_length)
                        labels = find_labels_within_threshold(midi_data, instrument_name, notes, current_sequence[-1][0].start, threshold)
                        all_sequences.append([instrument_name, genre, current_sequence, labels])
                        current_sequence.pop(0)

                        note_index -= 1

                note_index += 1

    
    return all_sequences

def find_related_note_amount(sequence):

    related_note_amount = 0

    for tuple in sequence:
        
        is_valid = tuple[1]

        if not is_valid:
            break
        
        related_note_amount += 1

    return related_note_amount


def find_labels_within_threshold(midi_data, current_sequence_instrument_name, current_sequence_instrument_notes, start_time, threshold, max_notes=3):
    """
    Finds the subsequent notes for each instrument within the threshold.
    Returns a list of subsequent notes for every instrument.
    If there are no subsequent notes within the threshold for a specific instrument, uses dummy notes.
    """
    labels = []

    labels.append((current_sequence_instrument_name, find_current_sequence_instrument_label(current_sequence_instrument_notes, start_time, threshold)))

    for instrument in midi_data.instruments:
        
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

        if instrument_name in instrument_list and instrument_name != current_sequence_instrument_name:
            notes = sorted(instrument.notes, key=lambda note: note.start)

            is_valid = True
            subsequent_notes = [(note, is_valid) 
                                for note in notes 
                                if start_time < note.start <= start_time + threshold][:max_notes]
            
            # Fill with dummy notes if fewer than max_notes are found
            while len(subsequent_notes) < max_notes:
                dummy_note = pretty_midi.Note(start = -1, end = -1, pitch = -1, velocity = -1)
                is_valid = False
                subsequent_notes.append((dummy_note, is_valid))
            
            labels.append((instrument_name, subsequent_notes))
    return labels

def find_current_sequence_instrument_label(current_sequence_instrument_notes, start_time, threshold, max_notes=3):

    notes = sorted(current_sequence_instrument_notes, key=lambda note: note.start)
    is_valid = True
    labels = [(note, is_valid) 
                        for note in notes 
                        if start_time < note.start <= start_time + threshold][:max_notes]
    
    # Fill with dummy notes if fewer than max_notes are found
    while len(labels) < max_notes:
        dummy_note = pretty_midi.Note(start = -1, end = -1, pitch = -1, velocity = -1)
        is_valid = False
        labels.append((dummy_note, is_valid))

    return labels
    

    
def apply_padding(sequence, sequence_length):

    while len(sequence) < sequence_length:

        dummy_note = pretty_midi.Note(start = -1, end = -1, pitch = -1, velocity = -1)
        is_valid = False

        sequence.append((dummy_note, is_valid))

    return sequence


def create_csv_from_dataset(dataset_path, output_csv_path, sequence_length, max_track_amoount_per_genre):
    all_data = []
    
    total_track_amount = 1

    for genre_folder in tqdm(os.listdir(dataset_path)):
        genre_folder_path = os.path.join(dataset_path, genre_folder)

        genre_track_amount = 1
        
        for midi_file_name in tqdm(os.listdir(genre_folder_path)):
            midi_file_path = os.path.join(genre_folder_path, midi_file_name)

            if(genre_track_amount > max_track_amoount_per_genre):
                break

            try:
                midi_data = pretty_midi.PrettyMIDI(midi_file_path)

            except: 
                print(f"Error processing {midi_file_path}")
                continue

            threshold = get_bar_as_threshold(midi_data)
            midi_data = process_midi_file(midi_data, genre_folder, sequence_length, threshold, minimum_related_note_amount = 2)
            all_data.extend(midi_data)

            genre_track_amount += 1
        
        total_track_amount += genre_track_amount
        logging.info(f"Created {genre_folder} genre folder with {genre_track_amount} tracks in it.")
    
    logging.info(f"Dataset converted with a total number of {total_track_amount} tracks.")

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['Instrument', 'Genre', 'Sequences', 'Labels'])
    df.to_csv(output_csv_path, index=False)


logging.basicConfig(filename='logs/modify_dataset.log', level=logging.INFO)

instrument_list = instrument_names.copy()
instrument_list.sort()

create_csv_from_dataset(dataset_path = "fixed_genre_classified_dataset", output_csv_path = "sequences_with_labels.csv", sequence_length = 5, max_track_amoount_per_genre = 100)