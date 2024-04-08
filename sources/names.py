instrument_names = { 0:"Acoustic Grand Piano", 1:"Bright Acoustic Piano", 2:"Electric Grand Piano", 18:"Rock Organ", 19:"Church Organ", 20:"Electric Piano 1",
    24:"Acoustic Guitar (nylon)", 25:"Acoustic Guitar (steel)", 26:"Electric Guitar (jazz)", 27:"Electric Guitar (clean)", 28:"Electric Guitar (muted)", 29:"Overdriven Guitar",
    30:"Distortion Guitar", 31:"Guitar Harmonics", 32:"Acoustic Bass", 33:"Electric Bass (finger)", 35:"Fretless Bass", 36:"Slap Bass 1", 39:"Synth Bass 2", 50:"Drum"
    }

essential_instrument_names = ["Bass", "Drum", "Guitar", "Piano"]
essential_instrument_names_dict = {0:"Bass", 1:"Drum", 2:"Guitar", 3:"Piano"}

genres = ["Blues", "Country", "Jazz", "Latin", "Pop", "Rock"]

program_numbers = {"Blues": [33, 50, 26, 19], "Country": [32, 50, 25, 0], "Jazz": [36, 50, 26, 20], "Latin": [32, 50, 24, 1], "Pop": [35, 50, 27, 0], "Rock": [33, 50, 30, 18]}


print(f"{instrument_names}")
