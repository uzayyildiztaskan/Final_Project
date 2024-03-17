import mido
from mido import MidiFile, MidiTrack, MetaMessage
import os
import logging
from tqdm import tqdm

def move_meta_messages_to_first_track(midi_path, output_path):
    midi_file = MidiFile(midi_path)
    first_track = MidiTrack()
    other_tracks = []

    for track in midi_file.tracks:

        new_track = MidiTrack()

        for msg in track:

            if msg.type in ['set_tempo', 'key_signature', 'time_signature']:

                first_track.append(msg)

            else:

                new_track.append(msg)
        other_tracks.append(new_track)

    new_midi_file = MidiFile()
    new_midi_file.tracks.append(first_track)
    new_midi_file.tracks.extend(other_tracks)

    new_midi_file.save(output_path)

logging.basicConfig(filename='../logs/fixed_classified_genre_dataset.log', level=logging.DEBUG)

corrupted_file_count = 0

for genre in tqdm(os.listdir("../genre_classified_dataset")):
    for track in os.listdir(f"../genre_classified_dataset/{genre}"):

        if not (os.path.exists(f"../fixed_genre_classified_dataset/{genre}")):
            os.makedirs(f"../fixed_genre_classified_dataset/{genre}")

        try:
            move_meta_messages_to_first_track(f"../genre_classified_dataset/{genre}/{track}", f"../fixed_genre_classified_dataset/{genre}/{track}")

        except:

            print(f"Error processing: ../genre_classified_dataset/{genre}/{track}")
            corrupted_file_count += 1
            continue

logging.info(f"Fixed the genre classified dataset\n{corrupted_file_count} corrupted files found!")