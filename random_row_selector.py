import pandas as pd
import random
from sources.names import essential_instrument_names, genres

sequences = []

all_seeds = "all_seeds = ["

file_content = ""

for genre in genres:
    
    random_sequence_instruments = []
    all_seeds += "["

    for instrument in essential_instrument_names:
        
        df = pd.read_csv(f'./processed_excel_data/{instrument}.csv')

        for index, row in df.iterrows():

            if random.random() <= 0.15 and row['Genre'] == genre:
                
                sequences.append((row['Sequence']))

        file_content += f"{genre.replace(' ', '_').lower()}_random_sequences_{instrument.lower()} = {sequences}\n\n"

file_content += "all_seeds = [[blues_random_sequences_bass, blues_random_sequences_drum, blues_random_sequences_guitar, blues_random_sequences_piano], [country_random_sequences_bass, country_random_sequences_drum, country_random_sequences_guitar, country_random_sequences_piano], [electronic_random_sequences_bass, electronic_random_sequences_drum, electronic_random_sequences_guitar, electronic_random_sequences_piano], [folk_random_sequences_bass, folk_random_sequences_drum, folk_random_sequences_guitar, folk_random_sequences_piano], [international_random_sequences_bass, international_random_sequences_drum, international_random_sequences_guitar, international_random_sequences_piano], [jazz_random_sequences_bass, jazz_random_sequences_drum, jazz_random_sequences_guitar, jazz_random_sequences_piano], [latin_random_sequences_bass, latin_random_sequences_drum, latin_random_sequences_guitar, latin_random_sequences_piano], [new_age_random_sequences_bass, new_age_random_sequences_drum, new_age_random_sequences_guitar, new_age_random_sequences_piano], [pop_random_sequences_bass, pop_random_sequences_drum, pop_random_sequences_guitar, pop_random_sequences_piano], [rock_random_sequences_bass, rock_random_sequences_drum, rock_random_sequences_guitar, rock_random_sequences_piano], [rap_random_sequences_bass, rap_random_sequences_drum, rap_random_sequences_guitar, rap_random_sequences_piano], [reggae_random_sequences_bass, reggae_random_sequences_drum, reggae_random_sequences_guitar, reggae_random_sequences_piano], [rnb_random_sequences_bass, rnb_random_sequences_drum, rnb_random_sequences_guitar, rnb_random_sequences_piano], [vocal_random_sequences_bass, vocal_random_sequences_drum, vocal_random_sequences_guitar, vocal_random_sequences_piano]]"

with open('./sources/random_sequences.py', 'w') as file:
    file.write(file_content)

