import pandas as pd
import random
from sources.names import essential_instrument_names, genres

file_content = ""

for genre in genres:
    
    for instrument in essential_instrument_names:
        
        random_sequence_instruments = []
        
        df = pd.read_csv(f'./processed_excel_data/{instrument}.csv')

        for index, row in df.iterrows():

            if random.random() <= 0.15 and row['Genre'] == genre:
                
                random_sequence_instruments.append((row['Sequence']))

        file_content += f"{genre.replace(' ', '_').lower()}_random_sequences_{instrument.lower()} = {random_sequence_instruments}\n\n"

file_content += "all_seeds = [[blues_random_sequences_bass, blues_random_sequences_drum, blues_random_sequences_guitar, blues_random_sequences_piano], [country_random_sequences_bass, country_random_sequences_drum, country_random_sequences_guitar, country_random_sequences_piano], [jazz_random_sequences_bass, jazz_random_sequences_drum, jazz_random_sequences_guitar, jazz_random_sequences_piano], [latin_random_sequences_bass, latin_random_sequences_drum, latin_random_sequences_guitar, latin_random_sequences_piano], [pop_random_sequences_bass, pop_random_sequences_drum, pop_random_sequences_guitar, pop_random_sequences_piano], [rock_random_sequences_bass, rock_random_sequences_drum, rock_random_sequences_guitar, rock_random_sequences_piano]]"

with open('./sources/random_sequences.py', 'w') as file:
    file.write(file_content)

