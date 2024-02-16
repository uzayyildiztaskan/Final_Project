import os
import numpy as np
from music21 import converter, note, stream, chord
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

def getNotes(dataset_folder):
    notes = []  # bu liste çıkarılan notaları tutması için oluşturuldu

    for folder in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, folder)
        for file in os.listdir(f'{dataset_folder}/{folder}'):     # os.listdir(dataset_folder) -> dosyaların listesini alır.
            if file.endswith('.mid') or file.endswith('.midi'):
                midi = converter.parse(os.path.join(folder_path, file))
                #   music21 kullanarak MIDI dosyasını açar ve bu dosyayı bir Stream nesnesine dönüştürür.
                #   Stream notalara, akorlara vs erişim sağlar.
                print(f'Parsing {file}')
                # print("Included instruments")
                # for part in midi.parts:
                #     instr = part.getInstrument()
                #     instrumentName = instr.instrumentName if instr.instrumentName else "Unknown Instrument"
                #     print(f"{instr}\n")
                    
                first_note_or_chord_found = False
                for element in midi.flat.notesAndRests:     # midi dosyasındaki tüm notaları içeren bir objeyi tarar

                    if not first_note_or_chord_found:
                        if isinstance(element, (note.Note, chord.Chord)):
                            first_note_or_chord_found = True
                        else:
                            continue

                    if isinstance(element, note.Rest):
                        notes.append("rest")
                    if isinstance(element, note.Note):  # her bir elementin nota olup olmadığına bakılır
                        notes.append(str(element.pitch))    # her notanın frekans ve isim bilgisi notes listesine eklenir
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.pitches))
    return notes

# VERİ HAZIRLAMA
dataset_folder = 'data'   # veri setinin bulunduğu klasör
notes = getNotes(dataset_folder)    # veri seti notalara dönüştürülür
#print(notes)

seq_length = 15    # modelin her bir girişinde işlenecek olan nota sayısı

note_to_int = {note: i for i, note in enumerate(sorted(set(notes)))}
# enumerate(sorted(set(notes))) -> sıralı unique notalar üzerinde indeksleme yapılır
# bu adımda her bir nota değerini onun indeksiyle eşleştiren bir sözlük oluşturulur
int_to_note = {i: note for note, i in note_to_int.items()}
# her bir indeksi karşılık gelen nota değeriyle eşleştiren bir sözlük

input_seq = []  # modelin eğitimi için kullanılacak giriş dizisi
output_seq = []  # modelin eğitimi için kullanılacak çıkış dizisi

# her iterasyonda bir giriş(seq_in) ve çıkış(seq_out) dizisi oluşturulur
for i in range(0, len(notes) - seq_length, 1):
    seq_in = notes[i:i + seq_length]
    seq_out = notes[i + seq_length]

    input_seq.append([note_to_int[char] for char in seq_in])  # sayısal versiyonuyla eklenir
    output_seq.append(note_to_int[seq_out])  # sayısal versiyonuyla eklenir

pattern_length = len(input_seq)  # giriş ve çıkış dizilerinin uzunluğu

input_seq = np.reshape(input_seq, (pattern_length, seq_length, 1))  # LSTM'e uygun olacak şekilde yapılandırılır
input_seq = input_seq / float(len(set(notes)))  # normalizasyon adımı, sinir ağı modelleri genellikle 0 ile 1
                                                # arasındaki değerlerle daha iyi çalışır

output_seq = np.array(output_seq)   # çıkış dizisi NumPy dizisine dönüştürülür

model = Sequential([
    LSTM(64, input_shape=(input_seq.shape[1], input_seq.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    Dropout(0,2),
    LSTM(64),
    Dropout(0,2),
    Dense(len(set(notes)), activation='softmax')    # softmax notalar arasında olasılık dağılımı oluşturur
])

# Dense katmanındaki nöron sayısı modelin çıkışını temsil eden notaların sayısına eşit olmalı
# pattern_length, modelin mimarisini belirlemede değil, veri setinin nasıl işleneceğini belirlemede kullanılır

# Öğrenme oranını değiştirerek adam optimizer'ını özelleştir
optimizer = Adam(learning_rate=0.005)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
model.fit(input_seq, output_seq, epochs=50, batch_size=64, verbose=1)
# sparse_categorical_crossentropy multi-class yapılarda sınıflandırma problemlerine yardımcı olur

model.summary()


# start_note müzik dizisinin başlangıcını temsil eder(string verilir) -> veri setinden seed verdim
# num_notes -> oluşan müzikteki toplam nota sayısı (random 500 atadım)
def generate_notes(model, start_note, note_to_int, int_to_note, seq_length, num_notes):
    generated_notes = []    # oluşturulan müzik notaları burada depolanır
    pattern = [note_to_int[char] for char in start_note]    # start_note'taki her nota için note_to_int'te karşılık
    # gelen sayısal indexi bulur ve bu indexleri içeren bir liste oluşturur
    # pattern: modelin tahmin yapması için kullandığımız giriş verisi

    for _ in range(num_notes):  # num_notes kadar döner
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        # model bir seferde bir örnek üzerinden tahmin yapar
        # modelin tahmin yapması için kull. gereken giriş verisi
        # her zaman adımında bir nota temsil edilir

        prediction = model.predict(prediction_input, verbose=0)
        # prediction olasılık dağılımlarını içerir, model.predict() giriş verisi üzeirnde tahmin yapar

        index = np.random.choice(len(prediction[0]), p=prediction[0])
        # len(prediction[0]): model kaç nota için tahmin yapar
        # p=prediction[0] modelin çıkışındaki her notanın tahmini olasılığını içerir

        result = int_to_note[index]
        generated_notes.append(result)
        # çıkan sonuç notaya dönüştürülüp result'a atandı ve generated_notes'a eklendi

        pattern.append(index)
        # tahmin edilen index pattern listesine eklenir
        pattern = pattern[1:]
        # listenin ilk elemanını çıkararak pattern güncellenir, böylece giriş için yeni bir seed oluştu

    return generated_notes

start_note = notes[0:seq_length]
generated_notes = generate_notes(model, start_note, note_to_int, int_to_note, seq_length, num_notes=500)
print(generated_notes)

def create_midi(output_notes, file_name):
    output_midi = stream.Stream()   # müzikal veriyi temsil eden Stream nesnesi

    for pattern in output_notes:    # output_notes içindeki her bir eleman için bakalım
        if ('.' in pattern) or pattern.isdigit():
            chord_notes = pattern.split('.')
            notes = []  # akordun notalaranı içerecek olan liste
            for current_note in chord_notes:    # akordaki her nota için döner
                new_note = note.Note(current_note)  # her nota için bir Note nesnesi oluşturur
                notes.append(new_note)
            new_chord = chord.Chord(notes)  # notalardan oluşan akord oluşur, aynı anda çalar
            output_midi.append(new_chord)

        else:
            if(pattern == "rest"):
                new_note = note.Rest()
            else:
                new_note = note.Note(pattern)
            output_midi.append(new_note)

    output_midi.write('midi', fp=f"{file_name}.mid")


create_midi(generated_notes, "generated_music_4")