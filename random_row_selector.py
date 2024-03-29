import pandas as pd
import random

df = pd.read_csv('data_test.csv')
sequences = []

counter = 0

for sequence in df['Sequence']:

    if counter < 5550:        
        counter += 1 
        continue

    if counter > 6623:
        break

    counter += 1

    if random.random() <= 0.25:

        sequences.append(sequence)

print(sequences)