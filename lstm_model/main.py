from itertools import islice
import numpy as np

from model import MidiModel
from input import note_batch_generator, bits_to_string, read_songs
from midi import create_and_save_midi_file
from random import shuffle

data_path = "out-off12-fpiano.txt"
time_len = 16
batch_len = 64

songs = list(read_songs(data_path))
shuffle(songs)

total_song_count = len(songs)
train_song_count = int(total_song_count * 0.8)
test_song_count = total_song_count - train_song_count
train_songs = songs[:train_song_count]
test_songs = songs[train_song_count:]


midi_model = MidiModel(0.05, time_len, batch_len)    
test_result = midi_model.train(500, train_songs, test_songs)

# midi_model = MidiModel(0.05, time_len, batch_len, path = "model/final/test7.h5")
# midi_model.save("model/final/test7.h5")

for (line_nr, song) in read_songs(data_path):
    notes = []
    for bits in islice(midi_model.predict(song[:time_len]), 8 * 106):
        # print(bits_to_string(bits))
        notes.append(bits)
    
    create_and_save_midi_file(notes, "model/songs/line_%d.mid" % line_nr)
